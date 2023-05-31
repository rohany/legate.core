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

#include "core/runtime/mlir.h"
#include "core/data/store.h"
#include "core/task/task.h"
#include "core/utilities/dispatch.h"
#include "core/utilities/nvtx_help.h"

#include "legion.h"

#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/ExecutionEngine/ObjectCache.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/IRTransformLayer.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"


#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/NVGPUToNVVM/NVGPUToNVVM.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPU.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Conversion/SCFToOpenMP/SCFToOpenMP.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/OptimizeForNVVM.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include <iostream>

#ifdef LEGATE_USE_CUDA
#define _MLIR_NVTX_RANGE(s) nvtx::Range auto_range(s);
#else
#define _MLIR_NVTX_RANGE(s)
#endif

namespace legate {

// TODO (rohany): Comment...
extern void mlir_cuda_runtime_api_force_linkage(void);


static std::optional<llvm::OptimizationLevel> mapToLevel(unsigned optLevel,
                                                   unsigned sizeLevel) {
  switch (optLevel) {
  case 0:
    return llvm::OptimizationLevel::O0;

  case 1:
    return llvm::OptimizationLevel::O1;

  case 2:
    switch (sizeLevel) {
    case 0:
      return llvm::OptimizationLevel::O2;

    case 1:
      return llvm::OptimizationLevel::Os;

    case 2:
      return llvm::OptimizationLevel::Oz;
    }
    break;
  case 3:
    return llvm::OptimizationLevel::O3;
  }
  return std::nullopt;
}

std::function<llvm::Error(llvm::Module *)>
makeOptimizingTransformer(unsigned optLevel, unsigned sizeLevel,
                                llvm::TargetMachine *targetMachine) {
  return [optLevel, sizeLevel, targetMachine](llvm::Module *m) -> llvm::Error {
    std::optional<llvm::OptimizationLevel> ol = mapToLevel(optLevel, sizeLevel);
    if (!ol) {
      assert(false);
    }
    llvm::LoopAnalysisManager lam;
    llvm::FunctionAnalysisManager fam;
    llvm::CGSCCAnalysisManager cgam;
    llvm::ModuleAnalysisManager mam;

    llvm::PipelineTuningOptions tuningOptions;
    tuningOptions.LoopUnrolling = true;
    tuningOptions.LoopInterleaving = true;
    // tuningOptions.LoopVectorization = true;
    // tuningOptions.SLPVectorization = true;

    llvm::PassBuilder pb(targetMachine, tuningOptions);

    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    pb.crossRegisterProxies(lam, fam, cgam, mam);

    llvm::ModulePassManager mpm;
    mpm.addPass(pb.buildPerModuleDefaultPipeline(*ol));
    mpm.run(*m, mam);
    return llvm::Error::success();
  };
}

MLIRRuntime::MLIRRuntime() {
  std::cout << "Initializing MLIRRuntime." << std::endl;
  // Register the necessary dialects and passes. This is a separate
  // step from _loading_ them, which will occur later.
  mlir::registerAllDialects(this->registry);
  mlir::registerAllPasses();
  mlir::registerBuiltinDialectTranslation(this->registry);
  mlir::registerLLVMDialectTranslation(this->registry);
  mlir::registerOpenMPDialectTranslation(this->registry);

  // Create the MLIRContext once all of the dialects and
  // passes have been registered.
  // TODO (rohany): This probably has to be moved around once
  //  we let legate libraries register their own dialects.
  // Disable threading so that we don't create threads that Realm
  // doesn't know about.
  this->context = std::make_unique<mlir::MLIRContext>(this->registry, mlir::MLIRContext::Threading::DISABLED);
  this->context->loadAllAvailableDialects();

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Starting here to re-create some of the MLIR execution engine infrastructure,
  // but doing it in a way that I have more control over what's going on, in particular
  // keeping the JIT alive between invocations. Most of this code is adapted from
  // https://github.com/llvm/llvm-project/blob/main/mlir/lib/ExecutionEngine/ExecutionEngine.cpp.
  // this->llvmContext = std::make_unique<llvm::LLVMContext>();
  // Initialize the target machine.
  // TODO (rohany): It's possible that I'll need a separate JIT or separate
  //  target machine for GPUs and CPUs, but we'll deal with that when we get there.
  auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
  assert(tmBuilderOrError);
  auto tmOrError = tmBuilderOrError->createTargetMachine();
  assert(tmOrError);
  this->targetMachine = std::move(tmOrError.get());
  this->objectCache = std::make_unique<SimpleObjectCache>();

  // Callback to create the object layer with symbol resolution to current
  // process and dynamically linked libraries.
  // auto objectLinkingLayerCreator = [&](llvm::orc::ExecutionSession &session,
  //                                      const llvm::Triple &tt) {
  //   auto objectLayer = std::make_unique<llvm::orc::RTDyldObjectLinkingLayer>(session, nullptr);
  //   // We'll use the shared libraries that should be linked into this process, so
  //   // we don't need to do anything more, as done in the MLIR ExecutionEngine.
  //   return objectLayer;
  // };

  auto compileFunctionCreator = [&](llvm::orc::JITTargetMachineBuilder jtmb)
      -> llvm::Expected<std::unique_ptr<llvm::orc::IRCompileLayer::IRCompiler>> {
    // We want optimized code, -O3.
    jtmb.setCodeGenOptLevel(llvm::CodeGenOpt::Level::Default);
    // TODO (rohany): Not currently including a cache, as I think we'll
    //  manage caching at a higher level of the system than this.
    return std::make_unique<llvm::orc::SimpleCompiler>(*this->targetMachine, this->objectCache.get());
  };

  auto dataLayout = this->targetMachine->createDataLayout();
  this->jit = std::move(llvm::cantFail(
    llvm::orc::LLJITBuilder()
     .setCompileFunctionCreator(compileFunctionCreator)
     // .setObjectLinkingLayerCreator(objectLinkingLayerCreator)
     .setDataLayout(dataLayout)
     .create()
  ));
  // Construct an optimizing transformer to pass to the JIT later
  // when we add a module to it.
  this->llvmOptTransformer = makeOptimizingTransformer(2, 0, nullptr /* targetMachine */);

  // Resolve symbols that are statically linked in the current process.
  llvm::orc::JITDylib &mainJD = this->jit->getMainJITDylib();
  mainJD.addGenerator(
      cantFail(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
          dataLayout.getGlobalPrefix())));
}

MLIRRuntime::~MLIRRuntime() {
  // TODO (rohany): I don't want to think about teardown of the MLIR things yet,
  //  but this can be something to do in the future.
}

std::unique_ptr<mlir::MLIRContext>& MLIRRuntime::getContext() {
  return this->context;
}

// std::unique_ptr<llvm::LLVMContext>& MLIRRuntime::getLLVMContext() {
//   return this->llvmContext;
// }

std::unique_ptr<llvm::TargetMachine>& MLIRRuntime::getTargetMachine() {
  return this->targetMachine;
}

std::function<llvm::Error(llvm::Module*)>& MLIRRuntime::getOptTransformer() {
  return this->llvmOptTransformer;
}

std::unique_ptr<llvm::orc::LLJIT>& MLIRRuntime::getJIT() {
  return this->jit;
}

std::unique_ptr<SimpleObjectCache>& MLIRRuntime::getObjectCache() {
  return this->objectCache;
}

int64_t MLIRRuntime::getNextJITID() {
  return this->jitFunctionCtr++;
}

void MLIRRuntime::dumpMLIR(mlir::Operation* op) {
  mlir::AsmState asmState(op, mlir::OpPrintingFlags(), nullptr /* locationMap */, &this->fallbackResourceMap);
  op->print(llvm::outs(), asmState);
  llvm::outs() << "\n";
}

void MLIRRuntime::dumpAllObjects() {
  this->objectCache->dumpAllObjectsToFile("legate_kernels");
}


static std::string makePackedFunctionName(llvm::StringRef name) {
  return "_mlir_" + name.str();
}

// For each function in the LLVM module, define an interface function that wraps
// all the arguments of the original function and all its results into an i8**
// pointer to provide a unified invocation interface.
// This function is taken from the MLIR::ExecutionEngine implementation.
static void packFunctionArguments(llvm::Module *module) {
  auto &ctx = module->getContext();
  llvm::IRBuilder<> builder(ctx);
  mlir::DenseSet<llvm::Function *> interfaceFunctions;
  for (auto &func : module->getFunctionList()) {
    if (func.isDeclaration()) {
      continue;
    }
    if (interfaceFunctions.count(&func)) {
      continue;
    }

    // Given a function `foo(<...>)`, define the interface function
    // `mlir_foo(i8**)`.
    auto *newType = llvm::FunctionType::get(
        builder.getVoidTy(), builder.getInt8PtrTy()->getPointerTo(),
        /*isVarArg=*/false);
    auto newName = makePackedFunctionName(func.getName());
    auto funcCst = module->getOrInsertFunction(newName, newType);
    llvm::Function *interfaceFunc = llvm::cast<llvm::Function>(funcCst.getCallee());
    interfaceFunctions.insert(interfaceFunc);

    // Extract the arguments from the type-erased argument list and cast them to
    // the proper types.
    auto *bb = llvm::BasicBlock::Create(ctx);
    bb->insertInto(interfaceFunc);
    builder.SetInsertPoint(bb);
    llvm::Value *argList = interfaceFunc->arg_begin();
    llvm::SmallVector<llvm::Value *, 8> args;
    args.reserve(llvm::size(func.args()));
    for (auto [index, arg] : llvm::enumerate(func.args())) {
      llvm::Value *argIndex = llvm::Constant::getIntegerValue(
          builder.getInt64Ty(), llvm::APInt(64, index));
      llvm::Value *argPtrPtr =
          builder.CreateGEP(builder.getInt8PtrTy(), argList, argIndex);
      llvm::Value *argPtr =
          builder.CreateLoad(builder.getInt8PtrTy(), argPtrPtr);
      llvm::Type *argTy = arg.getType();
      argPtr = builder.CreateBitCast(argPtr, argTy->getPointerTo());
      llvm::Value *load = builder.CreateLoad(argTy, argPtr);
      args.push_back(load);
    }

    // Call the implementation function with the extracted arguments.
    llvm::Value *result = builder.CreateCall(&func, args);

    // Assuming the result is one value, potentially of type `void`.
    if (!result->getType()->isVoidTy()) {
      llvm::Value *retIndex = llvm::Constant::getIntegerValue(
          builder.getInt64Ty(), llvm::APInt(64, llvm::size(func.args())));
      llvm::Value *retPtrPtr =
          builder.CreateGEP(builder.getInt8PtrTy(), argList, retIndex);
      llvm::Value *retPtr =
          builder.CreateLoad(builder.getInt8PtrTy(), retPtrPtr);
      retPtr = builder.CreateBitCast(retPtr, result->getType()->getPointerTo());
      builder.CreateStore(result, retPtr);
    }

    // The interface function returns void.
    builder.CreateRetVoid();
  }
}

static void setupTargetTripleAndDataLayout(llvm::Module *llvmModule,
                                                     llvm::TargetMachine *tm) {
  llvmModule->setDataLayout(tm->createDataLayout());
  llvmModule->setTargetTriple(tm->getTargetTriple().getTriple());
}

MLIRModule::MLIRModule(
  mlir::OwningOpRef<mlir::ModuleOp> module,
  const std::string& kernelName,
  const std::vector<CompileTimeStoreDescriptor>& inputs,
  const std::vector<CompileTimeStoreDescriptor>& outputs,
  const std::vector<CompileTimeStoreDescriptor>& reducs
  ) : module_(std::move(module)), kernelName_(kernelName), inputs_(inputs), outputs_(outputs), reducs_(reducs) {}

void MLIRModule::lowerToLLVMDialect(MLIRRuntime* runtime, LegateVariantCode code) {
  _MLIR_NVTX_RANGE("lowerToLLVM")

  auto ctx = runtime->getContext().get();
  mlir::PassManager pm(ctx, this->module_.get()->getName().getStringRef(), mlir::PassManager::Nesting::Implicit);

  // TODO (rohany): Eventually, libraries will be able to register custom lowering
  //  hooks into these places as well. It's unclear how the ordering will work there
  //  though...
  if (code == LegateVariantCode::LEGATE_CPU_VARIANT) {
    mlir::ConvertVectorToLLVMPassOptions opts{};
    opts.x86Vector = true;
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertVectorToSCFPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createLowerAffinePass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertSCFToCFPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::memref::createExpandStridedMetadataPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::arith::createArithExpandOpsPass());
    pm.addPass(mlir::createConvertVectorToLLVMPass(opts));
    pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(mlir::createConvertMathToLLVMPass());
    pm.addPass(mlir::createConvertMathToLibmPass());
    // pm.addPass(mlir::createConvertVectorToLLVMPass(opts));
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  } else if (code == LegateVariantCode::LEGATE_OMP_VARIANT) {
#ifdef LEGATE_USE_OPENMP
    // TODO (rohany): I need to get this to work with vectorization too.
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createLowerAffinePass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::arith::createArithExpandOpsPass());
    pm.addPass(mlir::createConvertSCFToOpenMPPass());
    pm.addPass(mlir::createConvertOpenMPToLLVMPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::memref::createExpandStridedMetadataPass());
    pm.addPass(mlir::createConvertMathToLLVMPass());
    pm.addPass(mlir::createConvertMathToLibmPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
#else
    assert(false);
#endif
  } else if (code == LegateVariantCode::LEGATE_GPU_VARIANT) {
#ifdef LEGATE_USE_CUDA
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createLowerAffinePass());
    // TODO (rohany): Not sure how to generalize this blocking...
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createParallelLoopTilingPass({256}));
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createSCFForLoopCanonicalizationPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createGpuMapParallelLoopsPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createParallelLoopToGpuPass());
    pm.addPass(mlir::createConvertSCFToCFPass());
    // We have to do lower-affine again because of the re-introduction
    // of affine operations by some of these mapping passes to GPU.
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createLowerAffinePass());
    pm.addPass(mlir::createGpuKernelOutliningPass());
    pm.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createLowerGpuOpsToNVVMOpsPass());
    pm.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createConvertNVGPUToNVVMPass());
    pm.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::NVVM::createOptimizeForTargetPass());
    pm.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createGpuSerializeToCubinPass("nvptx64-nvidia-cuda", "sm_60", "+ptx60"));
    pm.addPass(mlir::createGpuToLLVMConversionPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::memref::createExpandStridedMetadataPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::arith::createArithExpandOpsPass());
    pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(mlir::createConvertMathToLLVMPass());
    pm.addPass(mlir::createConvertMathToLibmPass());
    // pm.addPass(mlir::createConvertVectorToLLVMPass(opts));
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    pm.addPass(mlir::createConvertIndexToLLVMPass());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
#else
    assert(false);
#endif
  } else {
    assert(false);
  }

  // TODO (rohany): Make this configurable to a log file ... 
  // pm.enableIRPrinting();

  if (mlir::failed(pm.run(this->module_.get()))) {
    std::cout << "Failed passes in lower to LLVM" << std::endl;
    runtime->dumpMLIR(this->module_.get());
    assert(false);
  }
}

uintptr_t MLIRModule::jitToLLVM(MLIRRuntime* runtime) {
  _MLIR_NVTX_RANGE("jitToLLVM")
  // Now, add the module to the JIT, invoke it, and return a function pointer to the generated code.
  std::unique_ptr<llvm::LLVMContext> llvmContext = std::make_unique<llvm::LLVMContext>();
  auto llvmModule = mlir::translateModuleToLLVMIR(this->module_.get(), *llvmContext, this->kernelName_);
  if (!llvmModule) {
    std::cout << "Failed translateModuletoLLVM: " << std::endl;
    runtime->dumpMLIR(this->module_.get());
  }
  assert(llvmModule);

  // TODO: Currently, the LLVM module created above has no triple associated
  //  with it. Instead, the triple is extracted from the TargetMachine, which is
  //  either based on the host defaults or command line arguments when specified
  //  (set-up by callers of this method). It could also be passed to the
  //  translation or dialect conversion instead of this.
  setupTargetTripleAndDataLayout(llvmModule.get(), runtime->getTargetMachine().get());
  // Construct a function that wraps the generated function for easy invocation.
  packFunctionArguments(llvmModule.get());

  llvm::orc::ThreadSafeModule tsm(std::move(llvmModule), std::move(llvmContext));
  llvm::cantFail(tsm.withModuleDo([&](llvm::Module& module) { return runtime->getOptTransformer()(&module); }));

  // TODO (rohany): We have to eventually delete this IR module as well.
  auto& jit = runtime->getJIT();
  llvm::cantFail(jit->addIRModule(std::move(tsm)));

  // Now get the function pointer out from the JIT. It won't compile
  // anything until we actually ask for a particular symbol.
  auto expectedSymbol = jit->lookup(makePackedFunctionName("_mlir_ciface_" + this->kernelName_));

  // TODO (rohany): Change this to take in the symbol name.
  // runtime->getObjectCache()->dumpToObjectFile("legate_kernel.o");

  // JIT lookup may return an Error referring to strings stored internally by
  // the JIT. If the Error outlives the ExecutionEngine, it would want have a
  // dangling reference, which is currently caught by an assertion inside JIT
  // thanks to hand-rolled reference counting. Rewrap the error message into a
  // string before returning. Alternatively, ORC JIT should consider copying
  // the string into the error message.
  if (!expectedSymbol) {
    std::string errorMessage;
    llvm::raw_string_ostream os(errorMessage);
    llvm::handleAllErrors(expectedSymbol.takeError(),
                          [&os](llvm::ErrorInfoBase &ei) { ei.log(os); });
    assert(false);
    return 0;
  }

  if (void *fptr = expectedSymbol->toPtr<void *>()) {
    return reinterpret_cast<uintptr_t>(fptr);
  }

  // TODO (rohany): Understand these failure cases.
  assert(false);
  return 0;
}

void MLIRModule::dump(MLIRRuntime* runtime) {
  runtime->dumpMLIR(this->module_.get());
}

/* static */
std::unique_ptr<MLIRModule> MLIRModule::fuseModules(
  MLIRRuntime* runtime,
  const std::string& kernelName,
  const std::vector<MLIRModule*>& modules,
  const std::vector<CompileTimeStoreDescriptor>& inputs,
  const std::vector<CompileTimeStoreDescriptor>& outputs,
  const std::vector<CompileTimeStoreDescriptor>& reducs,
  const std::map<int64_t, int32_t>& storeIDToIndexMapping
) {
  _MLIR_NVTX_RANGE("fuseModules")

  auto ctx = runtime->getContext().get();

  mlir::OpBuilder builder(ctx);
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToEnd(module->getBody());
  auto loc = mlir::NameLoc::get(mlir::StringAttr::get(ctx, "fused_task"));

  llvm::SmallVector<mlir::Type, 8> funcTypeArgs;
  for (auto store : inputs) { funcTypeArgs.push_back(buildMemRefType(ctx, store)); }
  for (auto store : outputs) { funcTypeArgs.push_back(buildMemRefType(ctx, store)); }
  for (auto store : reducs) { funcTypeArgs.push_back(buildMemRefType(ctx, store)); }
  auto funcType = builder.getFunctionType(funcTypeArgs, std::nullopt);
  // TODO (rohany): I should make this a static attribute or something, because it's used everywhere.
  mlir::NamedAttribute namedAttr(mlir::StringAttr::get(ctx, "llvm.emit_c_interface"), mlir::UnitAttr::get(ctx));
  // TODO (rohany): There's some trickiness I need to figure out here because I want the module to
  //  be "always valid", as in I could always lower it to LLVM and execute it, but I need temporary
  //  names for the functions I create as temporaries. Another option is to just writes passes that
  //  consume and modify this function, rather than creating new ones?
  auto func = builder.create<mlir::func::FuncOp>(loc, kernelName, funcType, std::vector<mlir::NamedAttribute>{namedAttr});
  auto block = func.addEntryBlock();
  builder.setInsertionPointToStart(block);

  mlir::IRMapping argumentMapper;
  for (auto mlirModule : modules) {
    auto& taskModule = mlirModule->module_;
    // There should only be a single function inside each of the modules.
    auto func = *taskModule->getOps<mlir::func::FuncOp>().begin();
    auto& funcBlocks = func.getBlocks();
    auto& header = funcBlocks.front();

    // Map all of the arguments of each task to the corresponding arguments
    // of the fused task.
    size_t index = 0;
    for (auto& store : mlirModule->inputs_) {
      argumentMapper.map(header.getArgument(index), block->getArgument(storeIDToIndexMapping.at(store.id)));
      index++;
    }
    for (auto& store : mlirModule->outputs_) {
      argumentMapper.map(header.getArgument(index), block->getArgument(storeIDToIndexMapping.at(store.id)));
      index++;
    }
    for (auto& store : mlirModule->reducs_) {
      argumentMapper.map(header.getArgument(index), block->getArgument(storeIDToIndexMapping.at(store.id)));
      index++;
    }

    // Copy all of the instructions from the task to the fused task.
    for (auto& funcBlock : funcBlocks) {
      for (auto& inst : funcBlock) {
        // We don't want to to include the return at the end of each task
        // in the resulting fused task.
        if (!mlir::isa<mlir::func::ReturnOp>(inst)) {
          builder.clone(inst, argumentMapper);
        }
      }
    }

    // Clear the argumentMapper for the next task.
    argumentMapper.clear();
  }
  // After all of the task bodies are copied over, include a return to terminate
  // the fused function.
  builder.create<mlir::func::ReturnOp>(loc);

  return std::make_unique<MLIRModule>(std::move(module), kernelName, inputs, outputs, reducs);
}

void MLIRModule::promoteTemporaryStores(
  MLIRRuntime* runtime,
  const std::vector<int32_t>& temporaryStoreOrdinals,
  const std::vector<int32_t>& resolutionOrdinalMapping
) {
  _MLIR_NVTX_RANGE("promoteTemps")
  auto ctx = runtime->getContext().get();
  mlir::PassManager pm(ctx, this->module_.get()->getName().getStringRef(), mlir::PassManager::Nesting::Implicit);
  mlir::OpPassManager& funcsPM = pm.nest<mlir::func::FuncOp>();
  funcsPM.addPass(std::make_unique<TemporaryStorePromotionPass>(temporaryStoreOrdinals, resolutionOrdinalMapping));
  if (mlir::failed(pm.run(this->module_.get()))) {
    assert(false);
  }

  // Remove the temporary stores from the module's metadata as well.
  std::set<int32_t> tempOrdinals(temporaryStoreOrdinals.begin(), temporaryStoreOrdinals.end());
  std::vector<CompileTimeStoreDescriptor> newInputs, newOutputs, newReducs;
  size_t idx = 0;
  for (size_t i = 0; i < this->inputs_.size(); i++) {
    if (tempOrdinals.count(idx) == 0) {
      newInputs.push_back(this->inputs_[i]);
    }
    idx++;
  }
  for (size_t i = 0; i < this->outputs_.size(); i++) {
    if (tempOrdinals.count(idx) == 0) {
      newOutputs.push_back(this->outputs_[i]);
    }
    idx++;
  }
  for (size_t i = 0; i < this->reducs_.size(); i++) {
    if (tempOrdinals.count(idx) == 0) {
      newReducs.push_back(this->reducs_[i]);
    }
    idx++;
  }
  this->inputs_ = newInputs;
  this->outputs_ = newOutputs;
  this->reducs_ = newReducs;
}

void MLIRModule::escalateIntermediateStorePrivilege(
  MLIRRuntime* runtime,
  const std::vector<int32_t>& intermediateStoreOrdinals,
  const std::vector<int32_t>& ordinalMapping
) {
  _MLIR_NVTX_RANGE("escalateIntermediates")
  auto ctx = runtime->getContext().get();
  mlir::PassManager pm(ctx, this->module_.get()->getName().getStringRef(), mlir::PassManager::Nesting::Implicit);
  mlir::OpPassManager& funcsPM = pm.nest<mlir::func::FuncOp>();
  funcsPM.addPass(std::make_unique<TemporaryStorePromotionPass>(intermediateStoreOrdinals, ordinalMapping));

  if (mlir::failed(pm.run(this->module_.get()))) {
    assert(false);
  }

  // Remove the temporary stores from this module's metadata.
  std::set<int32_t> tempOrdinals(intermediateStoreOrdinals.begin(), intermediateStoreOrdinals.end());
  std::vector<CompileTimeStoreDescriptor> newInputs;
  size_t idx = 0;
  for (size_t i = 0; i < this->inputs_.size(); i++) {
    if (tempOrdinals.count(idx) == 0) {
      newInputs.push_back(this->inputs_[i]);
    }
    idx++;
  }
  this->inputs_ = newInputs;
}

void MLIRModule::optimize(MLIRRuntime* runtime, LegateVariantCode code) {
  _MLIR_NVTX_RANGE("optimize")
  auto ctx = runtime->getContext().get();

  mlir::PassManager pm(ctx, this->module_.get()->getName().getStringRef(), mlir::PassManager::Nesting::Implicit);
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::affine::createLoopFusionPass(0, 0, true, mlir::affine::FusionMode::Greedy));
  // TODO (rohany): Investigate how many of these are needed...
  pm.addNestedPass<mlir::func::FuncOp>(mlir::affine::createAffineScalarReplacementPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::affine::createAffineScalarReplacementPass());

  // TODO (rohany): Eventually move this out of the "optimize" function.
  // TODO (rohany): Why did I write this? Maybe it was because this is a correctness
  //  thing that the core is doing to normalize memrefs, not an optimization thing.
  pm.addPass(mlir::memref::createNormalizeMemRefsPass());

  pm.addNestedPass<mlir::func::FuncOp>(std::make_unique<MemrefDimensionAccessNormalizingPass>());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::affine::createAffineLoopInvariantCodeMotionPass());
  // TODO (rohany): Investigate how many of these are needed...
  pm.addNestedPass<mlir::func::FuncOp>(mlir::affine::createAffineScalarReplacementPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::affine::createAffineScalarReplacementPass());

  // TODO (rohany): Until I can get vectorization and parallelization to work, I'm
  //  going to separate this out into a target-specific optimization.
  if (code == LegateVariantCode::LEGATE_CPU_VARIANT) {
    // TODO (rohany): Get vector size from CMake.
    std::vector<int64_t> vectorSizes(1, 4);
    mlir::affine::AffineVectorizeOptions options;
    options.vectorSizes = vectorSizes;
    pm.addNestedPass<mlir::func::FuncOp>(mlir::affine::createAffineVectorize(options));
  } else {
    pm.addNestedPass<mlir::func::FuncOp>(mlir::affine::createAffineParallelizePass());
  }

  // Some of the other passes can introduce some extra IR that is easy to
  // remove with some CSE.
  pm.addPass(mlir::createCSEPass());

  if (mlir::failed(pm.run(this->module_.get()))) {
    assert(false);
  }
}

MLIRTaskBodyGenerator::~MLIRTaskBodyGenerator() {}

CompileTimeStoreDescriptor::CompileTimeStoreDescriptor() : ndim(0), typ(Type::Code::INVALID), id(0), transform(nullptr) {}

CompileTimeStoreDescriptor::CompileTimeStoreDescriptor(
  int32_t ndim,
  legate_core_type_code_t typ,
  int64_t id,
  std::shared_ptr<TransformStack> transform
) : ndim(ndim), typ(static_cast<Type::Code>(typ)), id(id), transform(transform) {}

/* static */
void MLIRTask::register_variant(std::string& name, int task_id, LegateVariantCode code) {
  auto runtime = Legion::Runtime::get_runtime();
  runtime->attach_name(task_id, name.c_str(), false /* mutable */, true /* local_only */);

  // Register only the variant that we were requested to.
  const char* varName = nullptr;
  auto procKind = Legion::Processor::Kind::NO_KIND;
  switch (code) {
    case LegateVariantCode::LEGATE_CPU_VARIANT: {
      varName = "CPU";
      procKind = Legion::Processor::Kind::LOC_PROC;
      break;
    }
    case LegateVariantCode::LEGATE_OMP_VARIANT: {
      varName = "OMP";
      procKind = Legion::Processor::Kind::OMP_PROC;
      break;
    }
    case LegateVariantCode::LEGATE_GPU_VARIANT: {
      varName = "GPU";
      procKind = Legion::Processor::Kind::TOC_PROC;
      break;
    }
    default: {
      assert(false);
      break;
    }
  };
  {
    Legion::TaskVariantRegistrar registrar(task_id, false /* global */, varName);
    registrar.add_constraint(Legion::ProcessorConstraint(procKind));
    registrar.set_leaf();
    constexpr auto wrapper = detail::legate_task_wrapper<MLIRTask::body>;
    runtime->register_task_variant(registrar, Legion::CodeDescriptor(wrapper), nullptr, 0, LEGATE_MAX_SIZE_SCALAR_RETURN, code);
  }
}

template<typename ACC, typename T, int N>
StridedMemRefType<T, N> accessorToMemRef(ACC acc, Legion::Rect<N> bounds) {
  StridedMemRefType<T, N> memref{};

  auto base = acc.ptr(bounds.lo);
  memref.basePtr = const_cast<T*>(base);
  memref.data = const_cast<T*>(base);
  memref.offset = 0;
  for (int i = 0; i < N; i++) {
    memref.sizes[i] = bounds.hi[i] - bounds.lo[i] + 1;
    memref.strides[i] = acc.accessor.strides[i] / sizeof(T);
  }
  return memref;
}

struct AccessorToMemrefDescAlloc {
  template<Type::Code CODE, int DIM>
  void* operator()(const Store& store, bool read) {
    using T = legate_type_of<CODE>;
    StridedMemRefType<T, DIM> memref;
    if (read) {
      auto acc = store.read_accessor<T, DIM>();
      memref = accessorToMemRef<decltype(acc), T, DIM>(acc, store.shape<DIM>());
    } else {
      auto acc = store.write_accessor<T, DIM>();
      memref = accessorToMemRef<decltype(acc), T, DIM>(acc, store.shape<DIM>());
    }
    auto argPtr = static_cast<StridedMemRefType<T, DIM>*>(malloc(sizeof(decltype(memref))));
    *argPtr = memref;
    return argPtr;
  }
};

struct FutureToMemrefDescAlloc {
  template<Type::Code CODE>
  void* operator()(const Store& store, bool read) {
    using T = legate_type_of<CODE>;
    StridedMemRefType<T, 1> memref;
    if (read) {
      auto acc = store.read_accessor<T, 1>();
      memref = accessorToMemRef<decltype(acc), T, 1>(acc, store.shape<1>());
    } else {
      auto acc = store.write_accessor<T, 1>();
      memref = accessorToMemRef<decltype(acc), T, 1>(acc, store.shape<1>());
    }
    auto argPtr = static_cast<StridedMemRefType<T, 1>*>(malloc(sizeof(decltype(memref))));
    *argPtr = memref;
    return argPtr;
  }
};

/* static */
void MLIRTask::body(TaskContext& context) {
  auto& inputs = context.inputs();
  auto& outputs = context.outputs();
  auto& reducs = context.reductions();
  auto& scalars = context.scalars();

  assert(reducs.size() == 0);
  // TODO (rohany): Not handling the calling convention of applications
  //  passing scalars to the tasks yet.
  assert(scalars.size() == 1);

  typedef void (*func_t) (void**);
  auto func = scalars[0].value<func_t>();

  // TODO (rohany): Generating the body would also allow us to not malloc
  //  a bunch of times on each task invocation...

  llvm::SmallVector<void*, 8> argData;
  llvm::SmallVector<void*, 8> dimDatas;
  llvm::SmallVector<void*, 8> dimArgData;
  // Pack all of the inputs and outputs into Memref types.
  auto addStoreToArgs = [&](Store& store, bool read) {
    // TODO (rohany): Comment...
    {
      StridedMemRefType<int64_t, 1> memref{};

      auto data = (int64_t*)malloc(sizeof(int64_t) * store.dim());
      memref.basePtr = data;
      memref.data = data;
      memref.offset = 0;
      memref.sizes[0] = store.dim();
      memref.strides[0] = 1;
      auto dom = store.domain();
      for (int i = 0; i < store.dim(); i++) {
        data[i] = dom.hi()[i] - dom.lo()[i] + 1;
      }

      auto argPtr = (StridedMemRefType<int64_t, 1>*)malloc(sizeof(StridedMemRefType<int64_t, 1>));
      *argPtr = memref;
      dimDatas.push_back(data);
      dimArgData.push_back(argPtr);
    }
    while (store.transformed()) {
      store.remove_transform();
    }
    void* argPtr = nullptr;
    if (store.is_future()) {
      argPtr = type_dispatch(store.code(), FutureToMemrefDescAlloc{}, store, read);
    } else {
      argPtr = double_dispatch(store.dim(), store.code(), AccessorToMemrefDescAlloc{}, store, read);
    }
    argData.push_back(argPtr);
  };

  for (auto& store : inputs) {
    addStoreToArgs(store, true);
  }
  for (auto& store : outputs) {
    addStoreToArgs(store, false);
  }
  for (auto arg : dimArgData) {
    argData.push_back(arg);
  }

  llvm::SmallVector<void*, 8> args;
  for (size_t i = 0; i < argData.size(); i++) {
    args.push_back(&argData[i]);
  }

  (*func)(args.data());

  for (auto it : dimDatas) {
    free(it);
  }

  // Free everything after the body executes.
  for (auto it : argData) {
    free(it);
  }
}

mlir::Type coreTypeToMLIRType(mlir::MLIRContext* ctx, Type::Code typ) {
  switch (typ) {
    case Type::Code::BOOL: {
      return mlir::IntegerType::get(ctx, 1, mlir::IntegerType::SignednessSemantics::Signless);
    }
    case Type::Code::INT8: {
      return mlir::IntegerType::get(ctx, 8, mlir::IntegerType::SignednessSemantics::Signed);
    }
    case Type::Code::INT16: {
      return mlir::IntegerType::get(ctx, 16, mlir::IntegerType::SignednessSemantics::Signed);
    }
    case Type::Code::INT32: {
      return mlir::IntegerType::get(ctx, 32, mlir::IntegerType::SignednessSemantics::Signed);
    }
    case Type::Code::INT64: {
      return mlir::IntegerType::get(ctx, 64, mlir::IntegerType::SignednessSemantics::Signed);
    }
    case Type::Code::UINT8: {
      return mlir::IntegerType::get(ctx, 8, mlir::IntegerType::SignednessSemantics::Unsigned);
    }
    case Type::Code::UINT16: {
      return mlir::IntegerType::get(ctx, 16, mlir::IntegerType::SignednessSemantics::Unsigned);
    }
    case Type::Code::UINT32: {
      return mlir::IntegerType::get(ctx, 32, mlir::IntegerType::SignednessSemantics::Unsigned);
    }
    case Type::Code::UINT64: {
      return mlir::IntegerType::get(ctx, 64, mlir::IntegerType::SignednessSemantics::Unsigned);
    }
    case Type::Code::FLOAT16: {
      return mlir::Float16Type::get(ctx);
    }
    case Type::Code::FLOAT32: {
      return mlir::Float32Type::get(ctx);
    }
    case Type::Code::FLOAT64: {
      return mlir::Float64Type::get(ctx);
    }
    case Type::Code::COMPLEX64: {
      // TODO (rohany): Check this...
      return mlir::ComplexType::get(mlir::Float32Type::get(ctx));
    }
    case Type::Code::COMPLEX128: {
      // TODO (rohany): Check this...
      return mlir::ComplexType::get(mlir::Float64Type::get(ctx));
    }
    case Type::Code::STRING: {
      assert(false);
      return mlir::Float16Type::get(ctx);
    }
    default:
      std::cout << "Typ: " << typ << std::endl;
      assert(false);
      return mlir::Float16Type::get(ctx);
  }
}

mlir::MemRefType buildMemRefType(mlir::MLIRContext* ctx, const CompileTimeStoreDescriptor& desc) {
  // Construct the components needed for the basic memref type.
  auto typ = coreTypeToMLIRType(ctx, desc.typ);
  llvm::SmallVector<int64_t, 4> dims;
  dims.reserve(desc.ndim);
  for (int32_t i = 0; i < desc.ndim; i++) {
    dims.push_back(mlir::ShapedType::kDynamic);
  }

  // Now use the transform stack to construct an affine map from the transformed store
  // to the physical layout of the store at runtime. We'll start with the an identity
  // initial map, and then apply transformations on it in reverse order to construct
  // the final mapping.
  auto affineMap = mlir::AffineMap::getMultiDimIdentityMap(desc.ndim, ctx);

  desc.transform->iter_transforms([&](StoreTransform* ptr) {
    // Apply the inverse transformation as specified by the input
    // transform to the result of the affine map.
    if (Promote* promote = dynamic_cast<Promote*>(ptr); promote != nullptr) {
      auto dim = promote->get_extra_dim();
      auto results = affineMap.getResults();

      // We have to handle the case of a single dimension that was promoted
      // separately, i.e. a scalar store promoted into a multi-dimensional
      // object separately from the case of just dropping a dimension.
      if (results.size() == 1) {
        assert(dim == 0);
        affineMap = mlir::AffineMap::get(affineMap.getNumDims(), affineMap.getNumSymbols(), mlir::getAffineConstantExpr(0, ctx), ctx);
      } else {
        // In the standard case, just drop the promoted dimension
        // from the affine map.
        llvm::SmallVector<mlir::AffineExpr, 4> newExpr;
        for (size_t i = 0; i < results.size(); i++) {
          if (i == dim) continue;
          newExpr.push_back(results[i]);
        }
        affineMap = mlir::AffineMap::get(affineMap.getNumDims(), affineMap.getNumSymbols(), newExpr, ctx);
      }
    } else if (Shift* shift = dynamic_cast<Shift*>(ptr); shift != nullptr) {
      // Right now, it seems like we don't have to do anything special when a shift
      // transform has been applied, for the following reasons. If any of these reasons
      // change, we might have to revisit this decision:
      // 1) Right now we generate 0 to dim loops, and assume that the memrefs point
      //    onto the base of the allocation.
      // 2) Because of 1), the accessor construction takes into account the shift
      //    because the region requirement is constructed based on the shift from
      //    the legate python side.
    } else if (Transpose* transpose = dynamic_cast<Transpose*>(ptr); transpose != nullptr) {
      auto& axes = transpose->get_axes();
      auto results = affineMap.getResults();
      llvm::SmallVector<mlir::AffineExpr, 4> newExpr(results.size());
      // We're inverting the transform here. Since these vectors should
      // be very small, we can do the dumb thing.
      for (size_t i = 0; i < results.size(); i++) {
        // Find the position where axes[j] == i;
        for (size_t j = 0; j < results.size(); j++) {
          if (axes[j] == i) {
            newExpr[i] = results[j];
            break;
          }
        }
      }
      affineMap = mlir::AffineMap::get(affineMap.getNumDims(), affineMap.getNumSymbols(), newExpr, ctx);
    } else {
      assert(false);
    }
  });

  return mlir::MemRefType::get(dims, typ, affineMap);
}

std::pair<llvm::SmallVector<mlir::Value, 4>, llvm::SmallVector<mlir::Value, 4>> loopBoundsFromVar(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value var, int32_t ndim) {
  // TODO (rohany): In the future, this could be templated on LEGATE_CORE_MAX_DIM (or whatever it's called).
  llvm::SmallVector<mlir::Value, 4> loopLBs, loopUBs;
  loopLBs.reserve(ndim);
  loopUBs.reserve(ndim);

  auto zeroIndex = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
  for (int32_t i = 0; i < ndim; i++) {
    loopLBs.push_back(zeroIndex);
    loopUBs.push_back(builder.create<mlir::memref::DimOp>(loc, var, int64_t(i)));
  }

  return {loopLBs, loopUBs};
}

}
