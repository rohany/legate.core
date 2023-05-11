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
#include "core/utilities/dispatch.h"
#include "core/task/task.h"

#include "legion.h"

#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/IRTransformLayer.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/TargetSelect.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include <iostream>

namespace legate {

MLIRRuntime::MLIRRuntime() {
  std::cout << "Initializing MLIRRuntime." << std::endl;
  // Register the necessary dialects and passes. This is a separate
  // step from _loading_ them, which will occur later.
  mlir::registerAllDialects(this->registry);
  mlir::registerAllPasses();
  mlir::registerBuiltinDialectTranslation(this->registry);
  mlir::registerLLVMDialectTranslation(this->registry);

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
    jtmb.setCodeGenOptLevel(llvm::CodeGenOpt::Level::Aggressive);
    // TODO (rohany): Not currently including a cache, as I think we'll
    //  manage caching at a higher level of the system than this.
    return std::make_unique<llvm::orc::SimpleCompiler>(*this->targetMachine, nullptr /* cache */);
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
  this->llvmOptTransformer = mlir::makeOptimizingTransformer(3, 0, nullptr /* targetMachine */);

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

int64_t MLIRRuntime::getNextJITID() {
  return this->jitFunctionCtr++;
}

void MLIRRuntime::dumpMLIR(mlir::Operation* op) {
  mlir::AsmState asmState(op, mlir::OpPrintingFlags(), nullptr /* locationMap */, &this->fallbackResourceMap);
  op->print(llvm::outs(), asmState);
  llvm::outs() << "\n";
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

void MLIRModule::lowerToLLVMDialect(MLIRRuntime* runtime) {
  // TODO (rohany): Eventually, this will need to accept input from individual
  //  libraries about how to lower things.
  auto ctx = runtime->getContext().get();
  mlir::LLVMConversionTarget convTarget(*ctx);
  convTarget.addLegalOp<mlir::ModuleOp>();
  mlir::LLVMTypeConverter typeConverter(ctx);
  mlir::RewritePatternSet patterns(ctx);
  mlir::populateAffineToStdConversionPatterns(patterns);
  mlir::populateSCFToControlFlowConversionPatterns(patterns);
  mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
  mlir::populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
  mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
  mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);
  // mlir::populateOpenMPToLLVMConversionPatterns(typeConverter, patterns);
  mlir::FrozenRewritePatternSet frozenPatterns(std::move(patterns));
  if (mlir::failed(mlir::applyFullConversion(this->module_.get(), convTarget, frozenPatterns))) {
    assert(false);
  }
}

uintptr_t MLIRModule::jitToLLVM(MLIRRuntime* runtime) {
  // Now, add the module to the JIT, invoke it, and return a function pointer to the generated code.
  std::unique_ptr<llvm::LLVMContext> llvmContext = std::make_unique<llvm::LLVMContext>();
  auto llvmModule = mlir::translateModuleToLLVMIR(this->module_.get(), *llvmContext);
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
  auto ctx = runtime->getContext().get();

  mlir::OpBuilder builder(ctx);
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToEnd(module->getBody());
  auto loc = mlir::NameLoc::get(mlir::StringAttr::get(ctx, "fused_task"));

  llvm::SmallVector<mlir::Type, 8> funcTypeArgs;
  for (auto store : inputs) { funcTypeArgs.push_back(buildMemRefTypeOfDim(ctx, store.ndim, store.typ)); }
  for (auto store : outputs) { funcTypeArgs.push_back(buildMemRefTypeOfDim(ctx, store.ndim, store.typ)); }
  for (auto store : reducs) { funcTypeArgs.push_back(buildMemRefTypeOfDim(ctx, store.ndim, store.typ)); }
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

void MLIRModule::optimize(MLIRRuntime* runtime) {
  auto ctx = runtime->getContext().get();
  mlir::PassManager pm(ctx, this->module_.get()->getName().getStringRef(), mlir::PassManager::Nesting::Implicit);
  mlir::OpPassManager& funcsPM = pm.nest<mlir::func::FuncOp>();
  funcsPM.addPass(mlir::createCSEPass());
  funcsPM.addPass(mlir::createLoopFusionPass(0, 0, true, mlir::FusionMode::Greedy));
  funcsPM.addPass(mlir::createAffineScalarReplacementPass());
  funcsPM.addPass(mlir::createAffineScalarReplacementPass());

  // pm.enableIRPrinting();

  if (mlir::failed(pm.run(this->module_.get()))) {
    assert(false);
  }
}

MLIRTaskBodyGenerator::~MLIRTaskBodyGenerator() {}

CompileTimeStoreDescriptor::CompileTimeStoreDescriptor() : ndim(0), typ(LegateTypeCode::MAX_TYPE_NUMBER), id(0) {}

CompileTimeStoreDescriptor::CompileTimeStoreDescriptor(int32_t ndim, LegateTypeCode typ, int64_t id) : ndim(ndim), typ(typ), id(id) {}

/* static */
void MLIRTask::register_variant(std::string& name, int task_id) {
  auto runtime = Legion::Runtime::get_runtime();
  runtime->attach_name(task_id, name.c_str(), false /* mutable */, true /* local_only */);
  // TODO (rohany): Just register the CPU variant for now.
  {
    Legion::TaskVariantRegistrar registrar(task_id, false /* global */, "CPU");
    registrar.add_constraint(Legion::ProcessorConstraint(Legion::Processor::Kind::LOC_PROC));
    registrar.set_leaf();
    constexpr auto wrapper = detail::legate_task_wrapper<MLIRTask::body>;
    runtime->register_task_variant(registrar, Legion::CodeDescriptor(wrapper), nullptr, 0, LEGATE_MAX_SIZE_SCALAR_RETURN, LegateVariantCode::LEGATE_CPU_VARIANT);
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
  template<LegateTypeCode CODE, int DIM>
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
  // Pack all of the inputs and outputs into Memref types.
  auto addStoreToArgs = [&](const Store& store, bool read) {
    // TODO (rohany): Not handling transforms right now.
    assert(!store.transformed());
    auto argPtr = double_dispatch(store.dim(), store.code(), AccessorToMemrefDescAlloc{}, store, read);
    argData.push_back(argPtr);
  };

  for (auto& store : inputs) {
    addStoreToArgs(store, true);
  }
  for (auto& store : outputs) {
    addStoreToArgs(store, false);
  }

  llvm::SmallVector<void*, 8> args;
  for (size_t i = 0; i < argData.size(); i++) {
    args.push_back(&argData[i]);
  }

  (*func)(args.data());

  // Free everything after the body executes.
  for (auto it : argData) {
    free(it);
  }
}

mlir::Type coreTypeToMLIRType(mlir::MLIRContext* ctx, LegateTypeCode typ) {
  switch (typ) {
    case LegateTypeCode::BOOL_LT: {
      // TODO (rohany): MLIR doesn't have boolean types???
      assert(false);
      return mlir::Float16Type::get(ctx);
    }
    case LegateTypeCode::INT8_LT: {
      return mlir::IntegerType::get(ctx, 8, mlir::IntegerType::SignednessSemantics::Signed);
    }
    case LegateTypeCode::INT16_LT: {
      return mlir::IntegerType::get(ctx, 16, mlir::IntegerType::SignednessSemantics::Signed);
    }
    case LegateTypeCode::INT32_LT: {
      return mlir::IntegerType::get(ctx, 32, mlir::IntegerType::SignednessSemantics::Signed);
    }
    case LegateTypeCode::INT64_LT: {
      return mlir::IntegerType::get(ctx, 64, mlir::IntegerType::SignednessSemantics::Signed);
    }
    case LegateTypeCode::UINT8_LT: {
      return mlir::IntegerType::get(ctx, 8, mlir::IntegerType::SignednessSemantics::Unsigned);
    }
    case LegateTypeCode::UINT16_LT: {
      return mlir::IntegerType::get(ctx, 16, mlir::IntegerType::SignednessSemantics::Unsigned);
    }
    case LegateTypeCode::UINT32_LT: {
      return mlir::IntegerType::get(ctx, 32, mlir::IntegerType::SignednessSemantics::Unsigned);
    }
    case LegateTypeCode::UINT64_LT: {
      return mlir::IntegerType::get(ctx, 64, mlir::IntegerType::SignednessSemantics::Unsigned);
    }
    case LegateTypeCode::HALF_LT: {
      return mlir::Float16Type::get(ctx);
    }
    case LegateTypeCode::FLOAT_LT: {
      return mlir::Float32Type::get(ctx);
    }
    case LegateTypeCode::DOUBLE_LT: {
      return mlir::Float64Type::get(ctx);
    }
    case LegateTypeCode::COMPLEX64_LT: {
      // TODO (rohany): Check this...
      return mlir::ComplexType::get(mlir::Float32Type::get(ctx));
    }
    case LegateTypeCode::COMPLEX128_LT: {
      // TODO (rohany): Check this...
      return mlir::ComplexType::get(mlir::Float64Type::get(ctx));
    }
    case LegateTypeCode::STRING_LT: {
      assert(false);
      return mlir::Float16Type::get(ctx);
    }
    default:
      std::cout << "Typ: " << typ << std::endl;
      assert(false);
      return mlir::Float16Type::get(ctx);
  }
}

mlir::MemRefType buildMemRefTypeOfDim(mlir::MLIRContext* ctx, int32_t ndim, LegateTypeCode typ) {
  llvm::SmallVector<int64_t, 4> dims;
  dims.reserve(ndim);
  for (int32_t i = 0; i < ndim; i++) {
    dims.push_back(mlir::ShapedType::kDynamic);
  }
  return mlir::MemRefType::get(dims, coreTypeToMLIRType(ctx, typ));
}

}