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

#pragma once

/*
 * To avoid extremely long compile times, this file must avoid being
 * included in other files at all costs. Forward declarations of the
 * types used here are necessary, because otherwise all of the downstream
 * includes will need to handle reading the large LLVM/MLIR headers. Therefore,
 * for each new type declared here, also forward-declare it in mlir_decls.h,
 * which will be used by header consumers of MLIR functionality in legate.
 */

//#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
// TODO (rohany): I don't know if this is the right include to use.
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"

#include "core/data/transform.h"
#include "core/runtime/context.h"
#include "core/runtime/mlir_passes.h"
#include "core/utilities/typedefs.h"

namespace legate {

// The MLIRRuntime manages all MLIR and LLVM related state
// for generating and transforming MLIR and LLVM programs.
class MLIRRuntime {
 public:
  MLIRRuntime();
  ~MLIRRuntime();
 public:
   // TODO (rohany): Maybe rename this to getMLIRContext();
   std::unique_ptr<mlir::MLIRContext>& getContext();
   // std::unique_ptr<llvm::LLVMContext>& getLLVMContext();
   std::unique_ptr<llvm::TargetMachine>& getTargetMachine();
   std::function<llvm::Error(llvm::Module*)>& getOptTransformer();
   std::unique_ptr<llvm::orc::LLJIT>& getJIT();
   std::unique_ptr<SimpleObjectCache>& getObjectCache();
   int64_t getNextJITID();

   void dumpMLIR(mlir::Operation* op);
   void dumpAllObjects();
 private:
  // MLIR related state.
  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::FallbackAsmResourceMap fallbackResourceMap;

  // TODO (rohany): Excluding the parseConfig, executionEngine and targetMachine for now.

  // LLVM related state.
  // TODO (rohany): Unsure what this one should look like. I suspect we might need
  //  different optimization passes etc when we are targeting different backends.
  std::function<llvm::Error(llvm::Module*)> llvmOptTransformer;
  // TODO (rohany): I don't think we are supposed to create llvmContext's a bit
  //  more freely than having just a singleton around.
  // std::unique_ptr<llvm::LLVMContext> llvmContext;
  std::unique_ptr<llvm::TargetMachine> targetMachine;
  std::unique_ptr<llvm::orc::LLJIT> jit;
  std::unique_ptr<SimpleObjectCache> objectCache;
  int64_t jitFunctionCtr = 0;
};

// MLIRModule is a wrapper around an mlir ModuleOp, mainly for
// simplifying interoperation with Python.
class MLIRModule {
  public:
    MLIRModule(
      mlir::OwningOpRef<mlir::ModuleOp> module,
      const std::string& kernelName,
      const std::vector<CompileTimeStoreDescriptor>& inputs,
      const std::vector<CompileTimeStoreDescriptor>& outputs,
      const std::vector<CompileTimeStoreDescriptor>& reducs
    );

    static std::unique_ptr<MLIRModule> fuseModules(
      MLIRRuntime* runtime,
      const std::string& kernelName,
      // TODO (rohany): I don't want to make this raw pointers, but Cython
      //  doesn't have support for reference_wrapper yet.
      const std::vector<MLIRModule*>& modules,
      const std::vector<CompileTimeStoreDescriptor>& inputs,
      const std::vector<CompileTimeStoreDescriptor>& outputs,
      const std::vector<CompileTimeStoreDescriptor>& reducs,
      const std::map<int64_t, int32_t>& storeIDToIndexMapping
    );
 public:
    // TODO (rohany): This is not the final view of what this function
    //  may look like, but it's something to keep pushing the protoype
    //  forward.
    void lowerToLLVMDialect(MLIRRuntime* runtime);
    void dump(MLIRRuntime* runtime);

    // TODO (rohany): Comment...
    void promoteTemporaryStores(
      MLIRRuntime* runtime,
      const std::vector<int32_t>& temporaryStoreOrdinals,
      const std::vector<int32_t>& resolutionOrdinalMapping
    );

    // escalarateIntermediateStorePrivilege removes intermediate stores
    // that are produced by tasks in the fused buffer from the inputs of
    // a module so that they don't get launched with read-write privilege.
    // Precisely, we promote the read-write privilege on the input stores
    // to be only write privilege.
    void escalateIntermediateStorePrivilege(
      MLIRRuntime* runtime,
      const std::vector<int32_t>& intermedateStoreOrdinals,
      const std::vector<int32_t>& ordinalMapping
    );

    // TODO (rohany): Run a couple passes on this module. In the future this
    //  will be opt in by different libraries, but for now we'll just have
    //  a couple pre-done passes.
    void optimize(MLIRRuntime* runtime);

    uintptr_t jitToLLVM(MLIRRuntime* runtime);
  private:
    mlir::OwningOpRef<mlir::ModuleOp> module_;
    std::string kernelName_;

    std::vector<CompileTimeStoreDescriptor> inputs_;
    std::vector<CompileTimeStoreDescriptor> outputs_;
    std::vector<CompileTimeStoreDescriptor> reducs_;
};
// Have to include this typdef to get around some Cython parsing
// of pointer-typed template parameters.
typedef MLIRModule* MLIRModulePtr;

// TODO (rohany): Comment...
class CompileTimeStoreDescriptor {
public:
  CompileTimeStoreDescriptor();
  CompileTimeStoreDescriptor(int32_t ndim, LegateTypeCode typ, int64_t id, std::shared_ptr<TransformStack>);
  int32_t ndim;
  LegateTypeCode typ;
  int64_t id;
  std::shared_ptr<TransformStack> transform;
};

// TODO (rohany): Comment...
class MLIRTaskBodyGenerator {
 public:
   // TODO (rohany): Have to figure out what parameters this will take in...
   virtual std::unique_ptr<MLIRModule> generate_body(
     MLIRRuntime* runtime,
     const std::string& kernelName,
     const std::vector<CompileTimeStoreDescriptor>& inputs,
     const std::vector<CompileTimeStoreDescriptor>& outputs,
     const std::vector<CompileTimeStoreDescriptor>& reducs,
     char* buffer,
     int32_t buflen
   ) = 0;
   virtual ~MLIRTaskBodyGenerator();
};

class MLIRTask {
public:
  // register_variant (possibly renamed) is used to register a particular
  // task variant of the MLIR task
  static void register_variant(std::string& name, int task_id);

  // static void cpu_variant(legate::TaskContext& context);
  // TODO (rohany): Add the other task variants once this starts to work.
public:
  static void body(legate::TaskContext& context);
};

// class TemporaryStorePromotionPass
//   : public mlir::PassWrapper<TemporaryStorePromotionPass, mlir::OperationPass<mlir::func::FuncOp>> {
//
//  public:
//   TemporaryStorePromotionPass(const std::vector<int64_t>& temporaryStoreIDs,
//                               const std::map<int64_t, int64_t>& shapeResolutionMapping,
//                               const std::map<int64_t, int32_t>& storeIDToIndexMapping);
//   // TODO (rohany): Add in dependent dialect registration?
//   void runOnOperation() final;
//  private:
//   const std::vector<int64_t>& temporaryStoreIDs_;
//   const std::map<int64_t, int64_t>& shapeResolutionMapping_;
//   const std::map<int64_t, int32_t>& storeIDToIndexMapping_;
// };
// Utility functions for developing MLIR task bodies.
// TODO (rohany): In the future, maybe these go to a different file.

mlir::Type coreTypeToMLIRType(mlir::MLIRContext* ctx, LegateTypeCode typ);
mlir::MemRefType buildMemRefType(mlir::MLIRContext* ctx, const CompileTimeStoreDescriptor& desc);
std::pair<llvm::SmallVector<mlir::Value, 4>, llvm::SmallVector<mlir::Value, 4>> loopBoundsFromVar(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value var, int32_t ndim);

}