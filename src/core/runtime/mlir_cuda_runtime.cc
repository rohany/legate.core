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
#include <unordered_map>
#include <mutex>

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
  // Realm will set the correct cuContext at the entry of the CUDA task,
  // so we don't need to do that sort of book-keeping on entry.

  // Construct per-processor CUmodule and lock maps to maintain
  // loaded modules per-processor. There may be task parallelism
  // between fused GPU kernels, so we have to protect this table
  // with a lock.
  static std::mutex lock_table[LEGION_MAX_NUM_PROCS];
  static std::unordered_map<void*, CUmodule> module_table[LEGION_MAX_NUM_PROCS];
  const auto proc = Legion::Processor::get_executing_processor();
  const auto proc_id = proc.id & (LEGION_MAX_NUM_PROCS - 1);
  CUmodule module = nullptr;
  {
    std::lock_guard<std::mutex> guard(lock_table[proc_id]);
    // Probe the table to see if we can find the module. If we don't find
    // it for the current GPU, then we have to actually make the call.
    auto it = module_table[proc_id].find(data);
    if (it == module_table[proc_id].end()) {
      CUDA_REPORT_IF_ERROR(cuModuleLoadData(&module, data));
      module_table[proc_id][data] = module;
    } else {
      module = it->second;
    }
  }

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

EXTERNAL_LINKAGE void* mgpuMemAlloc(uint64_t sizeBytes, CUstream /*stream*/) {
  // We'll allocate on deferred buffers separate from the CUDA streams.
  // This function only handles device-side allocations, so we can always
  // do GPU_FB_MEM.
  auto buffer = legate::create_buffer<uint8_t>(sizeBytes, Legion::Memory::Kind::GPU_FB_MEM);
  return reinterpret_cast<void*>(buffer.ptr(0));
}

EXTERNAL_LINKAGE void mgpuMemcpy(void *dst, void *src, size_t sizeBytes, CUstream stream) {
  CUDA_REPORT_IF_ERROR(cuMemcpyAsync(reinterpret_cast<CUdeviceptr>(dst),
                                     reinterpret_cast<CUdeviceptr>(src),
                                     sizeBytes,
                                     stream));
}

}


