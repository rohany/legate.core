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

#include <stdio.h>

#include "cuda.h"
#include "legion.h"

#include "core/cuda/stream_pool.h"

#define EXTERNAL_LINKAGE  __attribute__((visibility("default")))


// This file contains overrides of MLIR's CUDA runtime to handle things
// in a Legate-specific way. In particular, we'll want to avoid doing duplicate
// cuModuleLoad operations, and also handle unloading of modules on our own.

#define CUDA_REPORT_IF_ERROR(expr)                                             \
  [](CUresult result) {                                                        \
    if (!result)                                                               \
      return;                                                                  \
    const char *name = nullptr;                                                \
    cuGetErrorName(result, &name);                                             \
    if (!name)                                                                 \
      name = "<unknown>";                                                      \
    fprintf(stderr, "'%s' failed with '%s'\n", #expr, name);                   \
  }(expr)

namespace legate {
  void mlir_cuda_runtime_api_force_linkage(void) {}
};

extern "C" {

EXTERNAL_LINKAGE CUmodule mgpuModuleLoad(void* data) {
  // TODO (rohany): Make a table per processor of maps from pointer
  //  to module so that we only load modules once. Just to check that
  //  this is working though, I'm going to start out without it.

  // Realm will set the correct cuContext at the entry of the CUDA task,
  // so we don't need to do anything before loading the module.
  CUmodule module = nullptr;
  CUDA_REPORT_IF_ERROR(cuModuleLoadData(&module, data));
  return module;
}

EXTERNAL_LINKAGE void mgpuModuleUnload(CUmodule) {
  // We're going to manage the lifetimes of our CUDA modules instead of
  // letting the generated code do it, so ignore the unload request.
}

CUfunction mgpuModuleGetFunction(CUmodule module, const char* name) {
  // We can just extract the function from the module as normal.
  CUfunction function = nullptr;
  CUDA_REPORT_IF_ERROR(cuModuleGetFunction(&function, module, name));
  return function;
}

EXTERNAL_LINKAGE void mgpuLaunchKernel(CUfunction function, intptr_t gridX, intptr_t gridY,
                 intptr_t gridZ, intptr_t blockX, intptr_t blockY,
                 intptr_t blockZ, int32_t smem, CUstream stream, void **params,
                 void **extra) {
  // Nothing special to do here, just launch the kernel.
  CUDA_REPORT_IF_ERROR(cuLaunchKernel(function, gridX, gridY, gridZ, blockX,
                              blockY, blockZ, smem, stream, params,
                              extra));
}

EXTERNAL_LINKAGE CUstream mgpuStreamCreate() {
  // Return the cached Legate stream for each GPU instead of creating a new one.
  return legate::cuda::StreamPool::get_stream_pool().get_stream().operator cudaStream_t();
}

EXTERNAL_LINKAGE void mgpuStreamDestroy(CUstream) {
 // Since we're managing the lifetimes of all of our streams, ignore the deletion
 // request from the generated code.
}

EXTERNAL_LINKAGE void mgpuStreamSynchronize(CUstream stream) {
  // Nothing special to do here, just synchronize the input stream.
  CUDA_REPORT_IF_ERROR(cuStreamSynchronize(stream));
}

}


