# Copyright 2023 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import cython
from typing import List

from libcpp cimport bool
from libcpp.map cimport map
from libcpp.memory cimport unique_ptr, shared_ptr, make_shared, make_unique
from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector
from libc.stdint cimport uintptr_t, int32_t, int64_t


cdef extern from "core/legate_c.h" nogil:
    ctypedef enum legate_core_variant_t:
        pass
    ctypedef enum legate_core_type_code_t:
        pass

# TODO (rohany): It doesn't seem like we can get around doing a bunch
#  of random copies etc at the boundary here.

cdef extern from "core/data/transform.h" namespace "legate" nogil:
    cdef cppclass StoreTransform:
        pass

    cdef cppclass TransformStack:
        TransformStack()
        TransformStack(unique_ptr[StoreTransform]&& transform,
                       shared_ptr[TransformStack]&& parent)
        unique_ptr[StoreTransform] pop()
        void dump()

    cdef cppclass Promote:
        Promote(int32_t, int64_t)

    cdef cppclass Shift:
        Shift(int32_t dim, int64_t offset)

    cdef cppclass Transpose:
        Tranpose(vector[int32_t]&& axes)


cdef class PyTransformStack:
    cdef shared_ptr[TransformStack] _stack
    def __init__(self):
        self._stack = make_shared[TransformStack]()

    def add_promote(self, object tx):
        cdef int extra_dim = tx._extra_dim
        cdef int dim_size = tx._dim_size
        self._stack = make_shared[TransformStack](
          make_unique[Promote](extra_dim, dim_size),
          move(self._stack)
        )

    def add_shift(self, object tx):
        cdef int dim = tx._dim
        cdef int offset = tx._offset
        self._stack = make_shared[TransformStack](
          make_unique[Shift](dim, offset),
          move(self._stack)
        )

    def add_transpose(self, object tx):
        cdef vector[int32_t] axes = vector[int32_t]()
        cdef int32_t axis
        for axis in tx._axes:
            axes.push_back(axis)
        self._stack = make_shared[TransformStack](
          make_unique[Transpose](move(axes)),
          move(self._stack)
        )

    def dump(self):
        self._stack.get().dump()


# TODO (rohany): I'm not sure how to import the definition from the other file...
#  I get various errors when I try to put these definitions into an MLIR specific file,
#  I'll leave this engineering simplification for another time.
cdef extern from "core/runtime/mlir.h" namespace "legate" nogil:
    cdef cppclass MLIRRuntime:
        int getNextJITID()
        void dumpAllObjects()

    cdef cppclass MLIRTask:
        @staticmethod
        void register_variant(string, int, legate_core_variant_t)

    cdef cppclass MLIRModule:
        void optimize(MLIRRuntime*, legate_core_variant_t)
        void lowerToLLVMDialect(MLIRRuntime*, legate_core_variant_t)
        uintptr_t jitToLLVM(MLIRRuntime*)

        void dump(MLIRRuntime*)
        @staticmethod
        unique_ptr[MLIRModule] fuseModules(
          MLIRRuntime*,
          const string&,
          const vector[MLIRModule*]& modules,
          const vector[CompileTimeStoreDescriptor]&,
          const vector[CompileTimeStoreDescriptor]&,
          const vector[CompileTimeStoreDescriptor]&,
          const map[int64_t, int32_t]&
        )
        void promoteTemporaryStores(
          MLIRRuntime*,
          const vector[int32_t]&,
          const vector[int32_t]&
        )
        void escalateIntermediateStorePrivilege(
          MLIRRuntime*,
          const vector[int32_t]&,
          const vector[int32_t]&
        )

    ctypedef MLIRModule* MLIRModulePtr

    cdef cppclass CompileTimeStoreDescriptor:
        CompileTimeStoreDescriptor()
        CompileTimeStoreDescriptor(int32_t, legate_core_type_code_t, int64_t, shared_ptr[TransformStack])
        int32_t ndim
        legate_core_type_code_t typ
        int64_t id


    cdef cppclass MLIRTaskBodyGenerator:
        unique_ptr[MLIRModule] generate_body(
          MLIRRuntime*,
          const string&,
          const vector[CompileTimeStoreDescriptor]&,
          const vector[CompileTimeStoreDescriptor]&,
          const vector[CompileTimeStoreDescriptor]&,
          char*,
          int32_t
        )

cdef extern from "core/runtime/context.h" namespace "legate" nogil:
    cdef cppclass LibraryContext:
        unsigned int get_task_id(long long)
        unsigned int get_max_preregistered_task_id()
        unsigned int get_mapper_id()
        int get_reduction_op_id(long long)
        unsigned int get_projection_id(long long)
        unsigned int get_sharding_id(long long)
        TaskInfo* find_task(long long)

cdef extern from "core/runtime/runtime.h" namespace "legate" nogil:
    cdef cppclass Runtime:
        @staticmethod
        Runtime* get_runtime()
        LibraryContext* find_library(string, bool)
        void initializeMLIRRuntime()
        MLIRRuntime* getMLIRRuntime()


cdef class PyCompileTimeStoreDescriptor:
    cdef CompileTimeStoreDescriptor desc

    def __init__(self, int ndim, legate_core_type_code_t typ, int id, PyTransformStack stack):
        self.desc = CompileTimeStoreDescriptor(ndim, typ, id, stack._stack)

    def __str__(self):
        return f"CompileTimeStoreDescriptor({self.desc.ndim}, {self.desc.typ}, {self.desc.id})"

    def __repr__(self):
        return self.__str__()

    @property
    def ndim(self) -> int:
        return self.desc.ndim

    @property
    def typ(self) -> legate_core_type_code_t:
        return self.desc.typ

    @property
    def id(self) -> int:
        return self.desc.id

    # The ID is something used during code generation to disambiguate
    # different stores while compiling an individual set of tasks. However,
    # I don't plan on using the ID as part of the fingerprint when deciding
    # if a particular trace can be replayed or something. Therefore, it is
    # excluded from the hash and equality functions.

    def __hash__(self) -> int:
        return hash((self.desc.ndim, self.desc.typ))

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, PyCompileTimeStoreDescriptor)
            and self.desc.ndim == other.ndim
            and self.desc.typ == other.typ
        )


cdef class PyMLIRModule:
    cdef unique_ptr[MLIRModule] _module

    @staticmethod
    cdef PyMLIRModule from_unique_ptr(unique_ptr[MLIRModule] module):
        cdef PyMLIRModule result = PyMLIRModule.__new__(PyMLIRModule)
        result._module = move(module)
        return result

    @staticmethod
    cdef PyMLIRModule generate_module(
        MLIRRuntime* runtime,
        shared_ptr[MLIRTaskBodyGenerator] gen,
        vector[CompileTimeStoreDescriptor]& inputs,
        vector[CompileTimeStoreDescriptor]& outputs,
        vector[CompileTimeStoreDescriptor]& reducs,
        char* buffer,
        int32_t buflen,
    ):
        cdef int funcID = runtime.getNextJITID()
        cdef str kernelName = "legateMLIRKernel" + str(funcID)
        cdef unique_ptr[MLIRModule] module = gen.get().generate_body(runtime, kernelName.encode(), inputs, outputs, reducs, buffer, buflen)
        return PyMLIRModule.from_unique_ptr(move(module))

    def lowerToLLVMDialect(self, int variantCode):
        runtime = Runtime.get_runtime().getMLIRRuntime()
        self._module.get().lowerToLLVMDialect(runtime, cython.cast(legate_core_variant_t, variantCode))

    def optimize(self, int variantCode) -> None:
        cdef MLIRRuntime* runtime = Runtime.get_runtime().getMLIRRuntime()
        self._module.get().optimize(runtime, cython.cast(legate_core_variant_t, variantCode))

    def dump(self):
        runtime = Runtime.get_runtime().getMLIRRuntime()
        self._module.get().dump(runtime)

    def jitToLLVM(self) -> uintptr_t:
        runtime = Runtime.get_runtime().getMLIRRuntime()
        return self._module.get().jitToLLVM(runtime)

    @staticmethod
    def construct_fused_module(
        list modules,
        list input_stores,
        list output_stores,
        list reduc_stores,
        dict store_id_to_index_mapping
    ) -> PyMLIRModule:
        cdef vector[MLIRModule*] module_ptrs = vector[MLIRModulePtr](len(modules))
        cdef vector[CompileTimeStoreDescriptor] inputs = vector[CompileTimeStoreDescriptor](len(input_stores))
        cdef vector[CompileTimeStoreDescriptor] outputs = vector[CompileTimeStoreDescriptor](len(output_stores))
        cdef vector[CompileTimeStoreDescriptor] reducs = vector[CompileTimeStoreDescriptor](len(reduc_stores))
        cdef map[int64_t, int32_t] store_id_to_index = map[int64_t, int32_t]();

        # Copy over all of the store descriptors into the C++ vectors.
        cdef int i
        for i in range(len(input_stores)):
          inputs[i] = (<PyCompileTimeStoreDescriptor?>input_stores[i]).desc
        for i in range(len(output_stores)):
          outputs[i] = (<PyCompileTimeStoreDescriptor?>output_stores[i]).desc
        for i in range(len(reduc_stores)):
          reducs[i] = (<PyCompileTimeStoreDescriptor?>reduc_stores[i]).desc
        for i in range(len(modules)):
          module_ptrs[i] = (<PyMLIRModule?>modules[i])._module.get()

        # Also convert the mapping into a C++ object.
        for id, idx in store_id_to_index_mapping.items():
          store_id_to_index[id] = idx

        # TODO (rohany): If this ends up being used in one more place, we'll extract
        #  it into a method somewhere else.
        cdef MLIRRuntime* runtime = Runtime.get_runtime().getMLIRRuntime()
        cdef int funcID = runtime.getNextJITID()
        cdef str kernelName = "legateMLIRKernel" + str(funcID)
        cdef unique_ptr[MLIRModule] result = MLIRModule.fuseModules(
          runtime,
          kernelName.encode(),
          module_ptrs,
          inputs,
          outputs,
          reducs,
          store_id_to_index
        )
        return PyMLIRModule.from_unique_ptr(move(result))

    def promoteTemporaryStores(
        self,
        list temporary_store_ordinals,
        list resolved_shape_ordinals
    ) -> None:
        cdef vector[int32_t] temporary_store_ordinals_ = vector[int32_t](len(temporary_store_ordinals))
        cdef vector[int32_t] resolved_shape_ordinals_ = vector[int32_t](len(resolved_shape_ordinals))
        cdef int i
        for i in range(len(temporary_store_ordinals)):
          temporary_store_ordinals_[i] = temporary_store_ordinals[i]
          resolved_shape_ordinals_[i] = resolved_shape_ordinals_[i]
        cdef MLIRRuntime* runtime = Runtime.get_runtime().getMLIRRuntime()
        self._module.get().promoteTemporaryStores(runtime, temporary_store_ordinals_, resolved_shape_ordinals_)

    def escalateIntermediateStorePrivilege(
        self,
        list intermediate_store_ordinals,
        list ordinal_mapping
    ) -> None:
        cdef vector[int32_t] intermediate_store_ordinals_ = vector[int32_t](len(intermediate_store_ordinals))
        cdef vector[int32_t] ordinal_mapping_ = vector[int32_t](len(ordinal_mapping))
        cdef int i
        for i in range(len(intermediate_store_ordinals)):
          intermediate_store_ordinals_[i] = intermediate_store_ordinals[i]
          ordinal_mapping_[i] = ordinal_mapping[i]
        cdef MLIRRuntime* runtime = Runtime.get_runtime().getMLIRRuntime()
        self._module.get().escalateIntermediateStorePrivilege(runtime, intermediate_store_ordinals_, ordinal_mapping_)


cdef class PyMLIRTask:
    @staticmethod
    def register_variant(str name, int id, int variantCode):
        MLIRTask.register_variant(name.encode(), id, cython.cast(legate_core_variant_t, variantCode))


cdef class PyMLIRTaskBodyGenerator:
    cdef shared_ptr[MLIRTaskBodyGenerator] _gen

    @staticmethod
    cdef PyMLIRTaskBodyGenerator from_shared_ptr(shared_ptr[MLIRTaskBodyGenerator] gen):
        cdef PyMLIRTaskBodyGenerator result = PyMLIRTaskBodyGenerator.__new__(PyMLIRTaskBodyGenerator)
        result._gen = gen
        return result

    def generate_body(self, list input_stores, list output_stores, list reduc_stores, bytes buffer, int buflen) -> PyMLIRModule:
        cdef vector[CompileTimeStoreDescriptor] inputs = vector[CompileTimeStoreDescriptor](len(input_stores))
        cdef vector[CompileTimeStoreDescriptor] outputs = vector[CompileTimeStoreDescriptor](len(output_stores))
        cdef vector[CompileTimeStoreDescriptor] reducs = vector[CompileTimeStoreDescriptor](len(reduc_stores))

        cdef int i
        for i in range(len(input_stores)):
          inputs[i] = (<PyCompileTimeStoreDescriptor?>input_stores[i]).desc
        for i in range(len(output_stores)):
          outputs[i] = (<PyCompileTimeStoreDescriptor?>output_stores[i]).desc
        for i in range(len(reduc_stores)):
          reducs[i] = (<PyCompileTimeStoreDescriptor?>reduc_stores[i]).desc

        cdef Runtime* runtime = Runtime.get_runtime()
        cdef MLIRRuntime* mlirRuntime = runtime.getMLIRRuntime()
        return PyMLIRModule.generate_module(mlirRuntime, self._gen, inputs, outputs, reducs, buffer, buflen)


cdef extern from "core/task/task_info.h" namespace "legate" nogil:
    cdef cppclass TaskInfo:
        bool has_variant(int)
        string name()
        bool has_mlir_variant()
        shared_ptr[MLIRTaskBodyGenerator] get_mlir_body_generator()


cdef class CppTaskInfo:
    cdef const TaskInfo* _task_info

    @staticmethod
    cdef CppTaskInfo from_ptr(const TaskInfo* p_task_info):
        cdef CppTaskInfo result = CppTaskInfo.__new__(CppTaskInfo)
        result._task_info = p_task_info
        return result

    @property
    def valid(self) -> bool:
        return self._task_info != NULL

    @property
    def name(self) -> str:
        return self._task_info.name()

    def has_variant(self, int variant_id) -> bool:
        return self._task_info.has_variant(
            cython.cast(legate_core_variant_t, variant_id)
        )

    def has_mlir_variant(self) -> bool:
        return self._task_info.has_mlir_variant()

    def get_mlir_body_generator(self) -> PyMLIRTaskBodyGenerator:
        gen = self._task_info.get_mlir_body_generator()
        return PyMLIRTaskBodyGenerator.from_shared_ptr(gen)


# It looks like we have to introduce a layer of indirection
# around the Runtime object because cdef cppclass objects
# cannot be used directly by Python code. So there has to
# be a Python class that interacts with the C++ for this
# to all work.
cdef class CppRuntime:
    @staticmethod
    def initializeMLIRRuntime():
        Runtime.get_runtime().initializeMLIRRuntime()

    @staticmethod
    def dumpMLIRObjects():
        Runtime.get_runtime().getMLIRRuntime().dumpAllObjects()


cdef class Context:
    cdef LibraryContext* _context

    def __cinit__(self, str library_name, bool can_fail=False):
        self._context = Runtime.get_runtime().find_library(library_name.encode(), can_fail)

    def get_task_id(self, long long local_task_id) -> int:
        return self._context.get_task_id(local_task_id)

    def get_max_preregistered_task_id(self) -> int:
        return self._context.get_max_preregistered_task_id()

    def get_mapper_id(self) -> int:
        return self._context.get_mapper_id()

    def get_reduction_op_id(self, long long local_redop_id) -> int:
        return self._context.get_reduction_op_id(local_redop_id)

    def get_projection_id(self, long long local_proj_id) -> int:
        return self._context.get_projection_id(local_proj_id)

    def get_sharding_id(self, long long local_shard_id) -> int:
        return self._context.get_sharding_id(local_shard_id)

    def find_task(self, long long local_task_id) -> CppTaskInfo:
        return CppTaskInfo.from_ptr(self._context.find_task(local_task_id))
