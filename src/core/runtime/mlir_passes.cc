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

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

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
    auto resolvedType = mlir::cast<mlir::MemRefType>(resolved.getType());
    auto dim = resolvedType.getRank();
    llvm::SmallVector<mlir::Value, 4> dims;
    for (size_t d = 0; d < dim; d++) {
      dims.push_back(builder.create<mlir::memref::DimOp>(loc, resolved, int64_t(d)));
    }
    // Use the computed dimensions to construct a new local allocation.
    auto newTemp = builder.create<mlir::memref::AllocOp>(loc, mlir::cast<mlir::MemRefType>(temporary.getType()), dims);
    // Finally eliminate the argument in favor of the local allocation.
    temporary.replaceAllUsesWith(newTemp);

    // TODO (rohany): Comment this up / implement it.

    // TODO (rohany): Get all of the uses of the new temporary that are a dim op.
    // TODO (rohany): For each use, get the dim op, get the constant, turn it into
    //  an integer, and then replace the use of the entire op with the corresponding
    //  dim op in the above vector.

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
  std::set<int32_t> tempOrdinals(this->temporaryStoreOrdinals_.begin(), this->temporaryStoreOrdinals_.end());
  size_t idx = 0;
  for (size_t i = 0; i < numArgs; i++) {
    if (tempOrdinals.count(idx) == 0) {
      newArgTypes.push_back(originalTypes[i]);
    }
    idx++;
  }
  auto newFuncType = builder.getFunctionType(newArgTypes, std::nullopt);
  op.setFunctionType(newFuncType);

  // TODO (rohany): We have to see what data structures are needed here,
  //  but this pass needs to do:
  //  1) Remove the arguments corresponding to temporary stores.
  //  2) For each removed temporary store, rewrite it as an allocation
  //     by matching it against a freshly allocated store with the dimensions
  //     of the store the temporary has been remapped to.

  // TODO (rohany): I think that I want to change the arguments here so that we
  //  are just dealing with argument ordinals, rather than store IDs, as the MLIR
  //  pass at this point has no concept of store IDs.
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
      auto newDim = builder.create<mlir::AffineLoadOp>(loc, newArg, dimOp.getIndex());
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

}
