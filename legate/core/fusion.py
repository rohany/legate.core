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

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Tuple,
)

from . import types as ty

from ._legion.util import BufferBuilder

if TYPE_CHECKING:
    from .operation import AutoTask, Task, Operation
    from .partition import PartitionBase
    from .solver import Strategy
    from .store import Store

# TODO (rohany): I plan on having runtime import this module,
#  so let's be careful to avoid circular imports.

# FusionConstraint is an abstract class that represents an extendable fusion
# constraint. Fusion constraints process tasks in a buffer one at a time and
# communicate whether it is semantically correct to fuse a buffer of tasks.
# The apply method may maintain internal state while processing task buffers.
class FusionConstraint:
    def apply(self, op: Operation, strat: Strategy) -> bool:
        pass


# AutoTaskConstraint checks whether the operation is indeed an AutoTask.
class AutoTaskConstraint(FusionConstraint):
    def apply(self, op: Operation, _: Strategy) -> bool:
        from .operation import AutoTask
        return isinstance(op, AutoTask)


# MLIRVariantConstraint checks that the given task has an MLIR variant.
class MLIRVariantConstraint(FusionConstraint):
    def apply(self, op: Operation, _: Strategy) -> bool:
        tid = op._task_id
        info = op.context._cpp_context.find_task(tid)
        assert(info.valid)
        return info.has_mlir_variant()


# MachineConstraint checks that the target machine for all operations in the
# considered window are the same.
class MachineConstraint(FusionConstraint):
    def __init__(self):
        self._machine = None

    def apply(self, op: Operation, _: Strategy) -> bool:
        if self._machine is None:
            self._machine = op.target_machine
            return True
        else:
            return self._machine == op.target_machine


# PartitionEquivalenceConstraint checks that each store is used in
# the same partition (i.e. the same view) across the window. If a store
# is viewed in multiple different ways, this implies communication, and
# thus no fusion.
class PartitionEquivalenceConstraint(FusionConstraint):
    def __init__(self):
        self._store_parts = {}

    def apply(self, op: Operation, strat: Strategy) -> bool:
        # TODO (rohany): Handle reductions.
        for input, sym in zip(op.inputs, op._input_parts):
            if not self._check(input, strat.get_partition(sym)):
                return False
        for output, sym in zip(op.outputs, op._output_parts):
            if not self._check(output, strat.get_partition(sym)):
                return False
        return True

    def _check(self, store: Store, part: PartitionBase) -> bool:
        if store not in self._store_parts:
            self._store_parts[store] = part
            return True
        else:
            return self._store_parts[store] == part


# FusionConstraintManager manages a set of user-provided fusion constraints,
# and controls how much of the task buffer can be fused.
class FusionConstraintManager:
    def __init__(self):
        self._constraints = []

    def register_constraint(self, constraint: FusionConstraint) -> None:
        self._constraints.append(constraint)

    # compute_fusable_prefix returns how large of a prefix of the input
    # list of operations may be fused together.
    def compute_fusable_prefix(self, ops: List[Operation], strategy: List[Strategy]) -> int:
        for idx, (op, strat) in enumerate(zip(ops, strategy)):
            for constraint in self._constraints:
                if not constraint.apply(op, strat):
                    return idx
        # If we made it here, the entire buffer is fusable.
        return len(ops)


class TaskWindowDescriptor:
    # Initialize TaskWindowDescriptor with a slice of operation.Task objects.
    # We can't do the import here because of circularity.
    def __init__(self, ops: List[Task]):
        from .launcher import ScalarArg

        # Need to check several things:
        # 1) The set of task IDs is the same.
        #    1b. TODO (rohany): Also need to check that the scalar arguments to the tasks are
        #        the same, for example binary ops... Have to think about how to
        #        do this easily.
        # 2) All input stores have same dimensions, types and transforms.
        # 3) The reference count status of each store should be the same (not the actual counts,
        #    but what was dropped versus what was held.
        # 4) The dependency graph of tasks and their store arguments need to be
        #    isomorphic. I realize now that it's not actually a graph, but a list
        #    of objects that we can determine an isomorphism. I think that we need
        #    to check for this isomorphism between both the stores and the storages
        #    as both relations are used by the fusion analysis.
        # 5) TODO (rohany): Have to also check for all stores that are futures with immediate values
        #    (like inline constants, not futures returned from other tasks), that those futures are the same.
        #    ACTUALLY, i think this is fine. That store will get added as input to the task as normal, and
        #    things will be OK since we know that it's a future, and we generate code that loads from the future!

        # Record a list of task IDs in the buffer.
        self.task_ids = []
        self.task_scalar_args = []

        # Record all of the store types, dimensions and transforms.
        self.store_types = []
        self.store_dims = []
        # TODO (rohany): Have to define transform stack equality and hashing.
        self.store_transforms = []
        self.store_liveness = []

        # Maintain a structure that mirrors the original layout of tasks and stores,
        # but remaps all of the store and storage IDs using a common naming scheme
        # so that the same pattern of stores and storages can be identified.
        store_idctr, storage_idctr = 0, 0
        store_ids, storage_ids = set(), set()
        self.store_generic_ids, self.storage_generic_ids = [], []
        # add_id adds the current id to result if id is not in seen,
        # and also remaps id to the value of counter. We use this to
        # check isomorphism between streams of task + store arguments.
        def add_id(id, counter, seen, result):
            if id in seen:
                return counter
            else:
                seen.add(id)
                result.append(counter)
                return counter + 1

        for op in ops:
            self.task_ids.append(op._task_id)

            # Pack task scalar arguments into comparable buffers.
            builder = BufferBuilder()
            for arg, dtype in op._scalar_args:
                ScalarArg(arg, dtype).pack(builder)
            self.task_scalar_args.append(bytes(builder.get_string()))

            op_store_inputs, op_store_outputs = [], []
            op_storage_inputs, op_storage_outputs = [], []
            for store in op.inputs:
                store_idctr = add_id(store._unique_id, store_idctr, store_ids, op_store_inputs)
                storage_idctr = add_id(store._storage._unique_id, storage_idctr, storage_ids, op_storage_inputs)
                self.store_types.append(store.type)
                self.store_dims.append(store.ndim)
                # self.store_transforms.append(store.transform)
                self.store_liveness.append(store.has_external_references())
            for store in op.outputs:
                store_idctr = add_id(store._unique_id, store_idctr, store_ids, op_store_outputs)
                storage_idctr = add_id(store._storage._unique_id, storage_idctr, storage_ids, op_storage_outputs)
                self.store_types.append(store.type)
                self.store_dims.append(store.ndim)
                # self.store_transforms.append(store.transform)
                self.store_liveness.append(store.has_external_references())

            # Convert the remapped store and storage lists to tuples so that
            # we can hash them later.
            self.store_generic_ids.append((tuple(op_store_inputs), tuple(op_store_outputs)))
            self.storage_generic_ids.append((tuple(op_storage_inputs), tuple(op_storage_outputs)))

        # Compute a hash up-front of this descriptor.
        self._hash = hash((
            hash(tuple(self.task_ids)),
            hash(tuple(self.task_scalar_args)),
            hash(tuple(self.store_types)),
            hash(tuple(self.store_dims)),
            hash(tuple(self.store_transforms)),
            hash(tuple(self.store_liveness)),
            hash(tuple(self.store_generic_ids)),
            hash(tuple(self.storage_generic_ids)),
        ))

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if not isinstance(other, TaskWindowDescriptor):
            return False

        return self.task_ids == other.task_ids and \
               self.task_scalar_args == other.task_scalar_args and \
               self.store_types == other.store_types and \
               self.store_dims == other.store_dims and \
               self.store_transforms == other.store_transforms and \
               self.store_liveness == other.store_liveness and \
               self.store_generic_ids == other.store_generic_ids and \
               self.storage_generic_ids == other.storage_generic_ids


# TODO (rohany): The idea here is to aggregate as much information as we need to be able
#  to map from the original tasks and partitioning scheme onto the new task and stores
#  for the fused task.
class FusedTaskConstructionDescriptor:
    def __init__(
            self,
            ops: List[Task],
            inputs: List[Store],
            outputs: List[Store],
            reducs: List[Store],
            taskid: int,
            funcptr: int,
    ):
        self.taskid = taskid
        self.funcptr = funcptr
        store_to_orig_pos = {}

        for opidx, op in enumerate(ops):
            for storeidx, store in enumerate(op.inputs):
                if store._unique_id not in store_to_orig_pos:
                    store_to_orig_pos[store._unique_id] = (opidx, storeidx)
            for storeidx, store in enumerate(op.outputs):
                if store._unique_id not in store_to_orig_pos:
                    store_to_orig_pos[store._unique_id] = (opidx, storeidx)
            # TODO (rohany): Handle reductions.

        self.inputs, self.outputs, self.reducs = [], [], []
        for store in inputs:
            self.inputs.append(store_to_orig_pos[store._unique_id])
        for store in outputs:
            self.outputs.append(store_to_orig_pos[store._unique_id])
        for store in reducs:
            self.reducs.append(store_to_orig_pos[store._unique_id])

    def build_task(self, ops: List[Task], strategies: List[Strategy]) -> Tuple[Task, Strategy]:
        # TODO (rohany): Worry about when these ops come from different libraries.
        newTask = ops[0].context.create_auto_task(self.taskid)
        self.add_stores_to_task(newTask, ops)
        self.add_impl_to_task(newTask)
        newStrat = self.build_new_strategy(newTask, ops, strategies)
        return newTask, newStrat

    def add_impl_to_task(self, fused: AutoTask):
        fused.add_scalar_arg(self.funcptr, ty.uint64)

    def add_stores_to_task(self, fused: AutoTask, ops: List[Task]):
        # TODO (rohany): Add reductions.
        for opidx, storeidx in self.inputs:
            op = ops[opidx]
            fused.add_input(op.inputs[storeidx])
        for opidx, storeidx in self.outputs:
            op = ops[opidx]
            fused.add_output(op.outputs[storeidx])

    # build_part_sym_mapping assumes that add_stores_to_task has already been called.
    def build_part_sym_mapping(self, fused: AutoTask, ops: List[Task]):
        part_sym_mapping = {}
        # TODO (rohany): I don't think we need to remap symbols that no longer
        #  appear in the final set of stores, which is what my original implementation
        #  is doing. I'm going to try again here and just remap the things that we need.
        for idx, (opidx, storeidx) in enumerate(self.inputs):
            op = ops[opidx]
            part_sym_mapping[op._input_parts[storeidx]] = fused._input_parts[idx]
        for idx, (opidx, storeidx) in enumerate(self.outputs):
            op = ops[opidx]
            part_sym_mapping[op._output_parts[storeidx]] = fused._output_parts[idx]
        for idx, (opidx, storeidx) in enumerate(self.reducs):
            op = ops[opidx]
            part_sym_mapping[op._reduction_parts[storeidx]] = fused._reduction_parts[idx]
        return part_sym_mapping

    def build_new_strategy(self, fused: AutoTask, ops: List[Task], strategies: List[Strategy]):
        # Merge all of the individual stratgies into a single strategy.
        new_strat = strategies[0]
        for strat in strategies[1:]:
            new_strat.merge(strat)
        # Remap all of the needed symbols over to the new symbols.
        return new_strat.remap(self.build_part_sym_mapping(fused, ops))
