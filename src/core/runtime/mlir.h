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

#include "core/runtime/context.h"
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
   int64_t getNextJITID();

   void dumpMLIR(mlir::Operation* op);
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
  int64_t jitFunctionCtr = 0;
};

// MLIRModule is a wrapper around an mlir ModuleOp, mainly for
// simplifying interoperation with Python.
class MLIRModule {
  public:
    MLIRModule(mlir::OwningOpRef<mlir::ModuleOp> module, const std::string& kernelName);
 public:
    // TODO (rohany): This is not the final view of what this function
    //  may look like, but it's something to keep pushing the protoype
    //  forward.
    void lowerToLLVMDialect(MLIRRuntime* runtime);
    void dump(MLIRRuntime* runtime);

    uintptr_t jitToLLVM(MLIRRuntime* runtime);
  private:
    mlir::OwningOpRef<mlir::ModuleOp> module_;
    std::string kernelName_;
};

// TODO (rohany): Comment...
class CompileTimeStoreDescriptor {
public:
  CompileTimeStoreDescriptor();
  CompileTimeStoreDescriptor(int32_t ndim, LegateTypeCode typ);
  int32_t ndim;
  LegateTypeCode typ;
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
     const std::vector<CompileTimeStoreDescriptor>& reducs
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


// Utility functions for developing MLIR task bodies.
// TODO (rohany): In the future, maybe these go to a different file.

mlir::Type coreTypeToMLIRType(mlir::MLIRContext* ctx, LegateTypeCode typ);
mlir::MemRefType buildMemRefTypeOfDim(mlir::MLIRContext* ctx, int32_t ndim, LegateTypeCode typ);

}