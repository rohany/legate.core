/* Copyright 2021-2022 NVIDIA Corporation
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

#include <memory>
#include <unordered_map>

#include "legion.h"
// Must be included after legion.h
#include "legate_defines.h"

#include "core/comm/communicator.h"
#include "core/mapping/machine.h"
#include "core/mapping/mapping.h"
#include "core/runtime/resource.h"
#include "core/runtime/runtime.h"
#include "core/task/return.h"
#include "core/task/task_info.h"
#include "core/utilities/typedefs.h"

/**
 * @file
 * @brief Class definitions for legate::LibraryContext and legate::TaskContext
 */

namespace legate {

class Store;
class Scalar;

class InvalidTaskIdException : public std::exception {
 public:
  InvalidTaskIdException(const std::string& library_name,
                         int64_t offending_task_id,
                         int64_t max_task_id);

 public:
  virtual const char* what() const throw();

 private:
  std::string error_message;
};

/**
 * @ingroup runtime
 * @brief A library context that provides APIs for registering components
 */
class LibraryContext {
 public:
  /**
   * @brief Creates a library context from a library name and a configuration.
   *
   * A library is registered to the runtime only upon the first construction
   * and the `config` object is referred to only when the registration happens.
   * All the following constructions of `LibraryContext` only retrieve the
   * metadata from the runtime without registration and ignore the `config`.
   *
   * @param library_name Library name
   * @param config Resource configuration for the library. If the library is already
   * registered, the value will be ignored.
   */
  LibraryContext(const std::string& library_name,
                 const ResourceConfig& config,
                 std::unique_ptr<mapping::Mapper> mapper);

 public:
  LibraryContext(const LibraryContext&) = delete;
  LibraryContext(LibraryContext&&)      = default;

 public:
  /**
   * @brief Returns the name of the library
   *
   * @return Library name
   */
  const std::string& get_library_name() const;

 public:
  Legion::TaskID get_task_id(int64_t local_task_id) const;
  Legion::TaskID get_max_preregistered_task_id() const;
  Legion::MapperID get_mapper_id() const { return mapper_id_; }
  Legion::ReductionOpID get_reduction_op_id(int64_t local_redop_id) const;
  Legion::ProjectionID get_projection_id(int64_t local_proj_id) const;
  Legion::ShardingID get_sharding_id(int64_t local_shard_id) const;

 public:
  int64_t get_local_task_id(Legion::TaskID task_id) const;
  int64_t get_local_reduction_op_id(Legion::ReductionOpID redop_id) const;
  int64_t get_local_projection_id(Legion::ProjectionID proj_id) const;
  int64_t get_local_sharding_id(Legion::ShardingID shard_id) const;

 public:
  bool valid_task_id(Legion::TaskID task_id) const;
  bool valid_reduction_op_id(Legion::ReductionOpID redop_id) const;
  bool valid_projection_id(Legion::ProjectionID proj_id) const;
  bool valid_sharding_id(Legion::ShardingID shard_id) const;

 public:
  /**
   * @brief Registers a library specific reduction operator.
   *
   * The type parameter `REDOP` points to a class that implements a reduction operator.
   * Each reduction operator class has the following structure:
   *
   * @code{.cpp}
   * struct RedOp {
   *   using LHS = ...; // Type of the LHS values
   *   using RHS = ...; // Type of the RHS values
   *
   *   static const RHS identity = ...; // Identity of the reduction operator
   *
   *   template <bool EXCLUSIVE>
   *   __CUDA_HD__ inline static void apply(LHS& lhs, RHS rhs)
   *   {
   *     ...
   *   }
   *   template <bool EXCLUSIVE>
   *   __CUDA_HD__ inline static void fold(RHS& rhs1, RHS rhs2)
   *   {
   *     ...
   *   }
   * };
   * @endcode
   *
   * Semantically, Legate performs reductions of values `V0`, ..., `Vn` to element `E` in the
   * following way:
   *
   * @code{.cpp}
   * RHS T = RedOp::identity;
   * RedOp::fold(T, V0)
   * ...
   * RedOp::fold(T, Vn)
   * RedOp::apply(E, T)
   * @endcode
   * I.e., Legate gathers all reduction contributions using `fold` and applies the accumulator
   * to the element using `apply`.
   *
   * Oftentimes, the LHS and RHS of a reduction operator are the same type and `fold` and  `apply`
   * perform the same computation, but that's not mandatory. For example, one may implement
   * a reduction operator for subtraction, where the `fold` would sum up all RHS values whereas
   * the `apply` would subtract the aggregate value from the LHS.
   *
   * The reduction operator id (`REDOP_ID`) can be local to the library but should be unique
   * for each opeartor within the library.
   *
   * Finally, the contract for `apply` and `fold` is that they must update the
   * reference atomically when the `EXCLUSIVE` is `false`.
   *
   * @tparam REDOP Reduction operator to register
   * @param redop_id Library-local reduction operator ID
   *
   * @return Global reduction operator ID
   */
  template <typename REDOP>
  int32_t register_reduction_operator(int32_t redop_id);

 public:
  void register_task(int64_t local_task_id, std::unique_ptr<TaskInfo> task_info);
  const TaskInfo* find_task(int64_t local_task_id) const;

 private:
  Legion::Runtime* runtime_;
  const std::string library_name_;
  ResourceIdScope task_scope_;
  ResourceIdScope redop_scope_;
  ResourceIdScope proj_scope_;
  ResourceIdScope shard_scope_;

 private:
  Legion::MapperID mapper_id_;
  std::unique_ptr<mapping::Mapper> mapper_;
  std::unordered_map<int64_t, std::unique_ptr<TaskInfo>> tasks_;
  ResourceConfig resource_config_;
};

/**
 * @ingroup task
 * @brief A task context that contains task arguments and communicators
 */
class TaskContext {
 public:
  TaskContext(const Legion::Task* task,
              const std::vector<Legion::PhysicalRegion>& regions,
              Legion::Context context,
              Legion::Runtime* runtime);

 public:
  /**
   * @brief Returns input stores of the task
   *
   * @return Vector of input stores
   */
  std::vector<Store>& inputs() { return inputs_; }
  /**
   * @brief Returns output stores of the task
   *
   * @return Vector of output stores
   */
  std::vector<Store>& outputs() { return outputs_; }
  /**
   * @brief Returns reduction stores of the task
   *
   * @return Vector of reduction stores
   */
  std::vector<Store>& reductions() { return reductions_; }
  /**
   * @brief Returns by-value arguments of the task
   *
   * @return Vector of scalar objects
   */
  std::vector<Scalar>& scalars() { return scalars_; }
  /**
   * @brief Returns communicators of the task
   *
   * If a task launch ends up emitting only a single point task, that task will not get passed a
   * communicator, even if one was requested at task launching time. Therefore, most tasks using
   * communicators should be prepared to handle the case where the returned vector is empty.
   *
   * @return Vector of communicator objects
   */
  std::vector<comm::Communicator>& communicators() { return comms_; }

 public:
  /**
   * @brief Indicates whether the task is parallelized
   *
   * @return true The task is a single task
   * @return false The task is one in a set of multiple parallel tasks
   */
  bool is_single_task() const;
  /**
   * @brief Indicates whether the task is allowed to raise an exception
   *
   * @return true The task can raise an exception
   * @return false The task must not raise an exception
   */
  bool can_raise_exception() const { return can_raise_exception_; }
  /**
   * @brief Returns the point of the task. A 0D point will be returned for a single task.
   *
   * @return The point of the task
   */
  DomainPoint get_task_index() const;
  /**
   * @brief Returns the task group's launch domain. A single task returns an empty domain
   *
   * @return The task group's launch domain
   */
  Domain get_launch_domain() const;

 public:
  const mapping::MachineDesc& machine_desc() const { return machine_desc_; }

 public:
  /**
   * @brief Makes all of unbound output stores of this task empty
   */
  void make_all_unbound_stores_empty();
  ReturnValues pack_return_values() const;
  ReturnValues pack_return_values_with_exception(int32_t index,
                                                 const std::string& error_message) const;

 private:
  std::vector<ReturnValue> get_return_values() const;

 private:
  const Legion::Task* task_;
  const std::vector<Legion::PhysicalRegion>& regions_;
  Legion::Context context_;
  Legion::Runtime* runtime_;

 private:
  std::vector<Store> inputs_, outputs_, reductions_;
  std::vector<Scalar> scalars_;
  std::vector<comm::Communicator> comms_;
  bool can_raise_exception_;
  mapping::MachineDesc machine_desc_;
};

}  // namespace legate

#include "core/runtime/context.inl"
