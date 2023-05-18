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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

#include <vector>
#include <map>

/*
 * This file is meant to hold declarations and definitions of objects
 * that end up requiring runtime-type information (RTTI). MLIR is default
 * compiled without RTTI, and that can result in missing typeinfo errors
 * when compiling. To make matters more complicated, Realm requires RTTI
 * to compile correctly! To break this dependency chain, we separate out
 * the pieces that need RTTI into a separate file that does not import
 * Legion/Realm types and compile it separately without RTTI (`-fno-rtti`).
 */

namespace legate {

class TemporaryStorePromotionPass
  : public mlir::PassWrapper<TemporaryStorePromotionPass, mlir::OperationPass<mlir::func::FuncOp>> {

 public:
  TemporaryStorePromotionPass(const std::vector<int32_t>& temporaryStoreOrdinals,
                              const std::vector<int32_t>& resolutionOrdinalMapping);
  // TODO (rohany): Add in dependent dialect registration?
  void runOnOperation() final;
 private:
  const std::vector<int32_t>& temporaryStoreOrdinals_;
  const std::vector<int32_t>& resolutionOrdinalMapping_;
};

class IntermediateStorePrivilegeEscalationPass
  : public mlir::PassWrapper<IntermediateStorePrivilegeEscalationPass, mlir::OperationPass<mlir::func::FuncOp>> {
 public:
  IntermediateStorePrivilegeEscalationPass(const std::vector<int32_t>& intermediateOrdinals,
                                           const std::vector<int32_t>& ordinalMapping);
  void runOnOperation() final;
 private:
  const std::vector<int32_t>& intermediateOrdinals_;
  const std::vector<int32_t>& ordinalMapping_;
};

class MemrefDimensionAccessNormalizingPass :
  public mlir::PassWrapper<MemrefDimensionAccessNormalizingPass, mlir::OperationPass<mlir::func::FuncOp>> {
  public:
   MemrefDimensionAccessNormalizingPass();
   void runOnOperation() final;
};

}
