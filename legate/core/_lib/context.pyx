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
from libcpp.memory cimport unique_ptr, shared_ptr
from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector
from libc.stdint cimport uintptr_t


cdef extern from "core/legate_c.h" nogil:
    ctypedef enum legate_core_variant_t:
        pass
    ctypedef enum legate_core_type_code_t:
        pass

# TODO (rohany): It doesn't seem like we can get around doing a bunch
#  of random copies etc at the boundary here.


# TODO (rohany): I'm not sure how to import the definition from the other file...
#  I get various errors when I try to put these definitions into an MLIR specific file,
#  I'll leave this engineering simplification for another time.
cdef extern from "core/runtime/mlir.h" namespace "legate" nogil:
    cdef cppclass MLIRRuntime:
        int getNextJITID()

    cdef cppclass MLIRTask:
        @staticmethod
        void register_variant(string, int)

    cdef cppclass MLIRModule:
        void lowerToLLVMDialect(MLIRRuntime*)
        void dump(MLIRRuntime*)
        uintptr_t jitToLLVM(MLIRRuntime*)

    cdef cppclass CompileTimeStoreDescriptor:
        CompileTimeStoreDescriptor()
        CompileTimeStoreDescriptor(int, legate_core_type_code_t)
        int ndim
        legate_core_type_code_t typ


    cdef cppclass MLIRTaskBodyGenerator:
        unique_ptr[MLIRModule] generate_body(
          MLIRRuntime*,
          const string&,
          const vector[CompileTimeStoreDescriptor]&,
          const vector[CompileTimeStoreDescriptor]&,
          const vector[CompileTimeStoreDescriptor]&
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

    def __init__(self, int ndim, legate_core_type_code_t typ):
        self.desc = CompileTimeStoreDescriptor(ndim, typ)

    def __str__(self):
        return f"CompileTimeStoreDescriptor({self.desc.ndim}, {self.desc.typ})"

    def __repr__(self):
        return self.__str__()

    @property
    def ndim(self) -> int:
        return self.desc.ndim

    @property
    def typ(self) -> legate_core_type_code_t:
        return self.desc.typ

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
        vector[CompileTimeStoreDescriptor]& reducs
    ):
        cdef int funcID = runtime.getNextJITID()
        cdef str kernelName = "legateMLIRKernel" + str(funcID)
        cdef unique_ptr[MLIRModule] module = gen.get().generate_body(runtime, kernelName.encode(), inputs, outputs, reducs)
        return PyMLIRModule.from_unique_ptr(move(module))

    def lowerToLLVMDialect(self):
        runtime = Runtime.get_runtime().getMLIRRuntime()
        self._module.get().lowerToLLVMDialect(runtime)

    def dump(self):
        runtime = Runtime.get_runtime().getMLIRRuntime()
        self._module.get().dump(runtime)

    def jitToLLVM(self) -> uintptr_t:
        runtime = Runtime.get_runtime().getMLIRRuntime()
        return self._module.get().jitToLLVM(runtime)


cdef class PyMLIRTask:
    @staticmethod
    def register_variant(str name, int id):
        MLIRTask.register_variant(name.encode(), id)


cdef class PyMLIRTaskBodyGenerator:
    cdef shared_ptr[MLIRTaskBodyGenerator] _gen

    @staticmethod
    cdef PyMLIRTaskBodyGenerator from_shared_ptr(shared_ptr[MLIRTaskBodyGenerator] gen):
        cdef PyMLIRTaskBodyGenerator result = PyMLIRTaskBodyGenerator.__new__(PyMLIRTaskBodyGenerator)
        result._gen = gen
        return result

    def generate_body(self, list input_stores, list output_stores, list reduc_stores) -> PyMLIRModule:
        cdef vector[CompileTimeStoreDescriptor] inputs = vector[CompileTimeStoreDescriptor](len(input_stores))
        cdef vector[CompileTimeStoreDescriptor] outputs = vector[CompileTimeStoreDescriptor](len(output_stores))
        cdef vector[CompileTimeStoreDescriptor] reducs = vector[CompileTimeStoreDescriptor](len(reduc_stores))

        # TODO (rohany): I don't see a way around the following kind of code:
        #  Because this is a "def", not a "cdef", we're interfacing with arbitrary
        #  Python code, with no typing gaurantees. Therefore, we can't use the fact
        #  that the PyCompileTimeStoreDescriptor has a CompileTimeStoreDescriptor
        #  inside of it, and just invoke the copy constructor. Instead, we have to
        #  reference all of the necessary fields ourselves, and use the constructor
        #  ourself here. This isn't the end of the world, as Cython will tell us
        #  when the Constructor changes, but it's annoying that we can't just say
        #  `inputs[i] = input_stores[i]`.
        cdef int i
        for i in range(len(input_stores)):
          desc = input_stores[i]
          inputs[i] = CompileTimeStoreDescriptor(desc.ndim, desc.typ)
        for i in range(len(output_stores)):
          desc = output_stores[i]
          outputs[i] = CompileTimeStoreDescriptor(desc.ndim, desc.typ)
        for i in range(len(reduc_stores)):
          desc = reduc_stores[i]
          reducs[i] = CompileTimeStoreDescriptor(desc.ndim, desc.typ)

        cdef Runtime* runtime = Runtime.get_runtime()
        cdef MLIRRuntime* mlirRuntime = runtime.getMLIRRuntime()

        # Construct the name for the next kernel.
        return PyMLIRModule.generate_module(mlirRuntime, self._gen, inputs, outputs, reducs)


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
