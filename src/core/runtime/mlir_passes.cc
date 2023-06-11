/* Copyright 2023 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include "llvm/ExecutionEngine/ObjectCache.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"

#include "core/runtime/mlir_passes.h"

#include <set>
#include <iostream>

namespace legate {

TemporaryStorePromotionPass::TemporaryStorePromotionPass(
  const std::vector<int32_t>& temporaryStoreOrdinals,
  const std::vector<int32_t>& resolutionOrdinalMapping
) : temporaryStoreOrdinals_(temporaryStoreOrdinals),
    resolutionOrdinalMapping_(resolutionOrdinalMapping) {
  assert(this->temporaryStoreOrdinals_.size() == this->resolutionOrdinalMapping_.size());
}

void TemporaryStorePromotionPass::runOnOperation() {
  auto& ctx = this->getContext();
  mlir::func::FuncOp op = this->getOperation();
  auto& entryBlock = op.getBlocks().front();
  auto numArgs = entryBlock.getNumArguments();
  auto loc = mlir::NameLoc::get(mlir::StringAttr::get(&ctx, "temporaryStoreRemoval"));

  // Collect the arguments to remove, and the arguments to remap the shapes to.
  std::vector<mlir::Value> temporaryStores(this->temporaryStoreOrdinals_.size());
  std::vector<mlir::Value> resolvedStores(this->resolutionOrdinalMapping_.size());
  llvm::BitVector argsToDelete(numArgs);
  for (size_t i = 0; i < this->temporaryStoreOrdinals_.size(); i++) {
    temporaryStores[i] = entryBlock.getArgument(this->temporaryStoreOrdinals_[i]);
    resolvedStores[i] = entryBlock.getArgument(this->resolutionOrdinalMapping_[i]);
    argsToDelete.set(this->temporaryStoreOrdinals_[i]);
  }
  llvm::SmallVector<mlir::Type, 8> originalTypes;
  for (size_t i = 0; i < numArgs; i++) {
    originalTypes.push_back(entryBlock.getArgument(i).getType());
  }

  mlir::OpBuilder builder(&ctx);
  builder.setInsertionPointToStart(&entryBlock);

  // For each temporary store, introduce a newly allocated memref using the dimensions
  // of the resolved store.
  for (size_t i = 0; i < temporaryStores.size(); i++) {
    // Extract the type of the resolved argument and assert that it
    // is indeed a memref type.
    auto temporary = temporaryStores[i];
    auto resolved = resolvedStores[i];
    auto temporaryType = mlir::cast<mlir::MemRefType>(temporary.getType());
    auto resolvedType = mlir::cast<mlir::MemRefType>(resolved.getType());
    assert(temporaryType.getRank() == resolvedType.getRank());
    auto dim = resolvedType.getRank();
    llvm::SmallVector<mlir::Value, 4> dims;
    for (size_t d = 0; d < dim; d++) {
      dims.push_back(builder.create<mlir::memref::DimOp>(loc, resolved, int64_t(d)));
    }
    // Use the computed dimensions to construct a new local allocation.
    auto newTemp = builder.create<mlir::memref::AllocOp>(loc, mlir::cast<mlir::MemRefType>(temporary.getType()), dims);
    // Finally eliminate the argument in favor of the local allocation.
    temporary.replaceAllUsesWith(newTemp);

    // Forward all dimension operations on the temporary store to
    // the corresponding dimension operation on the forwarded store.
    // We don't want dimension operations on the temporaries laying
    // around because we want to encourage deletion of the temporaries.
    for (mlir::Operation* user : newTemp->getUsers()) {
      if (!mlir::isa<mlir::memref::DimOp>(user)) { continue; }
      mlir::memref::DimOp dimOp = mlir::cast<mlir::memref::DimOp>(user);
      auto idxOpt = dimOp.getConstantIndex();
      if (idxOpt) {
        auto idx = *idxOpt;
        dimOp.replaceAllUsesWith(dims[idx]);
      }
    }
  }
  // After all of the temporary arguments have been replaced, remove them from
  // the argument list.
  entryBlock.eraseArguments(argsToDelete);

  // Finally adjust the function type to exclude the replaced arguments.
  llvm::SmallVector<mlir::Type, 8> newArgTypes;
  for (size_t i = 0; i < entryBlock.getNumArguments(); i++) {
    newArgTypes.push_back(entryBlock.getArgument(i).getType());
  }
  auto newFuncType = builder.getFunctionType(newArgTypes, std::nullopt);
  op.setFunctionType(newFuncType);
}

IntermediateStorePrivilegeEscalationPass::IntermediateStorePrivilegeEscalationPass(
  const std::vector<int32_t>& intermediateOrdinals,
  const std::vector<int32_t>& ordinalMapping
) : intermediateOrdinals_(intermediateOrdinals), ordinalMapping_(ordinalMapping) {}

void IntermediateStorePrivilegeEscalationPass::runOnOperation() {
  auto& ctx = this->getContext();
  mlir::func::FuncOp op = this->getOperation();
  auto& entryBlock = op.getBlocks().front();
  auto numArgs = entryBlock.getNumArguments();
  auto loc = mlir::NameLoc::get(mlir::StringAttr::get(&ctx, "intermediateStorePrivilegeEscalation"));
  mlir::OpBuilder builder(&ctx);

  std::vector<mlir::Value> intermediateStores(this->intermediateOrdinals_.size());
  std::vector<mlir::Value> resolvedStores(this->ordinalMapping_.size());
  llvm::BitVector argsToDelete(numArgs);
  for (size_t i = 0; i < this->intermediateOrdinals_.size(); i++) {
    intermediateStores[i] = entryBlock.getArgument(this->intermediateOrdinals_[i]);
    resolvedStores[i] = entryBlock.getArgument(this->ordinalMapping_[i]);
    argsToDelete.set(this->intermediateOrdinals_[i]);
  }

  // Replace uses of the escalated ordinals with the remapped ordinal.
  for (size_t i = 0; i < intermediateStores.size(); i++) {
    auto intermediate = intermediateStores[i];
    auto resolved = resolvedStores[i];
    intermediate.replaceAllUsesWith(resolved);
  }
  entryBlock.eraseArguments(argsToDelete);

  // Finally adjust the function type to exclude the replaced arguments.
  llvm::SmallVector<mlir::Type, 8> newArgTypes;
  std::set<int32_t> tempOrdinals(this->intermediateOrdinals_.begin(), this->intermediateOrdinals_.end());
  size_t idx = 0;
  for (size_t i = 0; i < numArgs; i++) {
    if (tempOrdinals.count(idx) == 0) {
      newArgTypes.push_back(op.getArgumentTypes()[i]);
    }
    idx++;
  }
  auto newFuncType = builder.getFunctionType(newArgTypes, std::nullopt);
  op.setFunctionType(newFuncType);
}

MemrefDimensionAccessNormalizingPass::MemrefDimensionAccessNormalizingPass() {}

void MemrefDimensionAccessNormalizingPass::runOnOperation() {
  auto& ctx = this->getContext();
  mlir::func::FuncOp op = this->getOperation();
  auto& entryBlock = op.getBlocks().front();
  auto numArgs = entryBlock.getNumArguments();
  auto loc = mlir::NameLoc::get(mlir::StringAttr::get(&ctx, "dimensionAccessNormalization"));
  mlir::OpBuilder builder(&ctx);

  // For each argument, we'll add a new argument that represents the dynamic
  // size of the memref before any transformations are applied, which will
  // be supplied by the task wrapper. Then, we'll re-write any operations that
  // try to access the dimensions of the argument memref, as they may get
  // re-written to be invalid once the NormalizeMemRefs pass completes.
  llvm::SmallVector<mlir::Type, 8> origArgTypes;
  llvm::SmallVector<mlir::Type, 8> newArgTypes;
  for (size_t i = 0; i < numArgs; i++) {
    // First, add the new argument.
    auto arg = entryBlock.getArgument(i);
    auto argType = mlir::cast<mlir::MemRefType>(arg.getType());
    auto newArgType = mlir::MemRefType::get({argType.getRank()}, mlir::IndexType::get(&ctx));
    auto newArg = entryBlock.addArgument(newArgType, loc);
    origArgTypes.push_back(argType);
    newArgTypes.push_back(newArgType);

    // Now replace all uses of the initial argument dims with accesses into
    // this separate memref of dims.
    std::vector<mlir::Operation*> toErase;
    for (mlir::Operation* user : arg.getUsers()) {
      if (!mlir::isa<mlir::memref::DimOp>(user)) { continue; }
      mlir::memref::DimOp dimOp = mlir::cast<mlir::memref::DimOp>(user);
      builder.setInsertionPoint(user);
      auto newDim = builder.create<mlir::affine::AffineLoadOp>(loc, newArg, dimOp.getIndex());
      user->replaceAllUsesWith(newDim);
      toErase.push_back(user);
    }
    for (auto op : toErase) { op->erase(); }
  }

  llvm::SmallVector<mlir::Type, 8> argTypes;
  for (auto it : origArgTypes) { argTypes.push_back(it); }
  for (auto it : newArgTypes) { argTypes.push_back(it); }
  auto newFuncType = builder.getFunctionType(argTypes, std::nullopt);
  op.setFunctionType(newFuncType);
}

GreedyLoopCollapsingPass::GreedyLoopCollapsingPass() {}

void GreedyLoopCollapsingPass::runOnOperation() {
  // The GreedyLoopCollapsingPass collapses multi-dimensional for loops into a single
  // loop nest so that it can be scheduled onto GPUs.
  auto& ctx = this->getContext();
  mlir::func::FuncOp op = this->getOperation();
  mlir::OpBuilder builder(&ctx);
  auto loc = mlir::NameLoc::get(mlir::StringAttr::get(&ctx, "greedyLoopCollapsing"));

  llvm::SmallVector<mlir::scf::ParallelOp, 4> parallelOps;
  auto& entryBlock = op.getBlocks().front();
  for (auto& inst : entryBlock) {
    if (!mlir::isa<mlir::scf::ParallelOp>(inst)) continue;
    // Go through all parallel for loops in the function, and try to collapse them
    // into a 1-D loop.
    auto pfor = mlir::cast<mlir::scf::ParallelOp>(inst);
    parallelOps.push_back(pfor);
  }
  for (auto pfor : parallelOps) {
    constexpr int maxNestedDim = 3;
    auto numLoops = pfor.getNumLoops();
    assert(numLoops <= maxNestedDim);
    // We don't need to do any collapsing of 1D loops.
    if (numLoops == 1) continue;
    std::vector<unsigned> dims;
    for (int i = 0; i < numLoops; i++) dims.push_back(i);
    llvm::SmallVector<decltype(dims), 1> ref;
    ref.push_back(dims);
    mlir::collapseParallelLoops(pfor, ref);
  }
}

void SimpleObjectCache::notifyObjectCompiled(const llvm::Module *m,
                                             llvm::MemoryBufferRef objBuffer) {
  cachedObjects[m->getModuleIdentifier()] = llvm::MemoryBuffer::getMemBufferCopy(
      objBuffer.getBuffer(), objBuffer.getBufferIdentifier());
}

std::unique_ptr<llvm::MemoryBuffer> SimpleObjectCache::getObject(const llvm::Module *m) {
  auto i = cachedObjects.find(m->getModuleIdentifier());
  if (i == cachedObjects.end()) {
    // LLVM_DEBUG(llvm::dbgs() << "No object for " << m->getModuleIdentifier()
    //                   << " in cache. Compiling.\n");
    return nullptr;
  }
  // LLVM_DEBUG(llvm::dbgs() << "Object for " << m->getModuleIdentifier()
  //                   << " loaded from cache.\n");
  return llvm::MemoryBuffer::getMemBuffer(i->second->getMemBufferRef());
}

void SimpleObjectCache::dumpToObjectFile(llvm::StringRef outputFilename) {
  // Set up the output file.
  std::string errorMessage;
  auto file = mlir::openOutputFile(outputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return;
  }

  // Dump the object generated for a single module to the output file.
  assert(cachedObjects.size() == 1 && "Expected only one object entry.");
  auto &cachedObject = cachedObjects.begin()->second;
  file->os() << cachedObject->getBuffer();
  file->keep();
}

void SimpleObjectCache::dumpAllObjectsToFile(llvm::StringRef filename) {
  for (auto& it : this->cachedObjects) {
    auto outfilename = filename.str() + it.first().str() + ".o";
    // Set up the output file.
    std::string errorMessage;
    auto file = mlir::openOutputFile(outfilename, &errorMessage);
    if (!file) {
      llvm::errs() << errorMessage << "\n";
      return;
    }
    file->os() << it.second->getBuffer();
    file->keep();
  }
}

bool SimpleObjectCache::isEmpty() { return cachedObjects.empty(); }

void handleJITLookupError(llvm::Expected<llvm::orc::ExecutorAddr>& symbol) {
  std::string errorMessage;
  llvm::raw_string_ostream os(errorMessage);
  llvm::handleAllErrors(symbol.takeError(),
                        [&os](llvm::ErrorInfoBase &ei) { ei.log(os); });
}

}
