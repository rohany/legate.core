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

# import cython
#
# cdef extern from "core/runtime/mlir.h" namespace "legate" nogil:
#     cdef cppclass MLIRModule:
#         pass
#
#     cdef cppclass MLIRTaskBodyGenerator:
#         MLIRModule* generate_body()
#
#
# cdef class PyMLIRTaskBodyGenerator:
#     cdef MLIRTaskBodyGenerator* _gen
#
#     @staticmethod
#     cdef PyMLIRTaskBodyGenerator from_ptr(const MLIRTaskBodyGenerator* gen):
#         cdef PyMLIRTaskBodyGenerator result = PyMLIRTaskBodyGenerator.__new__(PyMLIRTaskBodyGenerator)
#         result._gen = gen
#         return result
#
#     cdef generate_body(self):
#         print("Generating MLIR Body")