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

from itertools import chain
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Set,
    Tuple,
)

from . import types as ty

from ._legion.util import BufferBuilder
from ._legion.future import Future

if TYPE_CHECKING:
    from .operation import AutoTask, Task, Operation
    from .partition import PartitionBase
    from .solver import Strategy
    from .store import Store, Storage
    from ._lib.context import PyMLIRModule  # type: ignore[import]

# TODO (rohany): I plan on having runtime import this module,
#  so let's be careful to avoid circular imports.

# FusionConstraint is an abstract class that represents an extendable fusion
# constraint. Fusion constraints process tasks in a buffer one at a time and
# communicate whether it is semantically correct to fuse a buffer of tasks.
# The apply method may maintain internal state while processing task buffers.
class FusionConstraint:
    def apply(self, op: Operation, strat: Strategy) -> bool:
        raise NotImplementedError


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


class LaunchSpaceEquivalenceConstraint(FusionConstraint):
    def __init__(self):
        self._launch_space = None
        self._launch_space_set = False
        self._single_ops = []

    # A single operation is promotable if all of the operands are futures.
    def is_single_op_promotable(self, op: Operation) -> bool:
        if len(op.reductions) > 0:
            # I can't imagine any single tasks launched with reduction
            # privilege. If I see this, I'm keeping the assert around
            # to investigate that operation more.
            assert(False)
            return False
        return all(s._storage.kind is Future for s in chain(op.inputs, op.outputs))

    def apply(self, op: Operation, strat: Strategy) -> bool:
        # We can always accept the first seen operation.
        if not self._launch_space_set:
            self._launch_space = strat.launch_domain
            self._launch_space_set = True
            return True

        # Maintain all of the single operations seen while the
        # launch space is None, since we need to check that all
        # of these operations are promotable in case we promote
        # the launch space to a non-None value.
        if self._launch_space is None and strat.launch_domain is None:
            self._single_ops.append(op)

        # Attempt to promote all previously seen single operations into
        # the new launch space.
        if self._launch_space is None and strat.launch_domain is not None:
            # If we are switching from a None to a non-None launch domain, all
            # of the operations we've seen so far with None launch domains must
            # be promotable.
            if all([self.is_single_op_promotable(op) for op in self._single_ops]):
                self._launch_space = strat.launch_domain
                return True
            else:
                return False

        # If we are maintaining a non-null launch space and see an
        # operation that isn't partitioned, see if that operation can
        # be promoted.
        if self._launch_space is not None and strat.launch_domain is None:
            return self.is_single_op_promotable(op)

        # Finally, just return if the launch spaces are equal at this point,
        # because either both are None or both are a shape.
        return self._launch_space == strat.launch_domain


# SupportedStoreTransformsConstraints ensures that all stores
# used by the operation are transformed in a way that the
# underlying code generator can handle.
class SupportedStoreTransformsConstraint(FusionConstraint):
    def apply(self, op: Operation, _: Strategy) -> bool:
        for input in op.inputs:
            if not input.transform.jit_supported():
                return False
        for output in op.outputs:
            if not output.transform.jit_supported():
                return False
        for reduc, _ in op.reductions:
            if not reduc.transform.jit_supported():
                return False
        return True


# ProducerConsumerViewConstraint checks that if we write to particular
# view of a store, then we can subsequently only perform reads from that
# same view of the store. Additionally, if we reduce to a store, we are
# not allowed to read from it at all. However, we are allowed to continue
# reducing to different views, as long as the same reduction function is
# being used. This constraint is waived for futures however, since different
# views and partitions on futures don't mean anything, as there is no such
# thing as partial aliasing of futures.
class ProducerConsumerViewConstraint(FusionConstraint):
    def __init__(self):
        self._storage_views: Dict[Storage, Tuple[Store, PartitionBase]] = {}
        self._storage_reductions: Dict[Storage, Tuple[Store, PartitionBase, int]] = {}

    def apply(self, op: Operation, strat: Strategy) -> bool:
        for input, sym in zip(op.inputs, op._input_parts):
            root = input._storage.get_root()
            part = strat.get_partition(sym)
            # TODO (rohany): I think we want a deeper check of equality here.
            if root in self._storage_views and self._storage_views[root] != (input, part) and root.kind != Future:
                return False
            # We cannot read any stores that are being reduced to.
            if root in self._storage_reductions:
                return False
        for output, sym in zip(op.outputs, op._output_parts):
            root = output._storage.get_root()
            part = strat.get_partition(sym)
            # TODO (rohany): I think we want a deeper check of equality here.
            if root in self._storage_views and self._storage_views[root] != (output, part):
                return False
            if root not in self._storage_views:
                self._storage_views[root] = (output, part)
        for (reduc, redop), sym in zip(op.reductions, op._reduction_parts):
            root = reduc._storage.get_root()
            part = strat.get_partition(sym)
            # TODO (rohany): I think we want a deeper check of equality here.
            if root in self._storage_reductions and self._storage_reductions[root] != (reduc, part, redop):
                return False
            if root not in self._storage_reductions:
                self._storage_reductions[root] = (reduc, part, redop)
        return True


# ReadAntiDependenceConstraint checks that if we read from multiple different
# views of a store, we do not write to any of those views. Reading from and
# writing to only one view of a store is OK.
class ReadAntiDependenceConstraint(FusionConstraint):
    def __init__(self):
        self._read_views: Dict[Storage, Set[Store, PartitionBase]] = {}

    def apply(self, op: Operation, strat: Strategy) -> bool:
        for input, sym in zip(op.inputs, op._input_parts):
            root = input._storage.get_root()
            part = strat.get_partition(sym)
            # Record this read view of the root storage.
            if root not in self._read_views:
                self._read_views[root] = set()
            self._read_views[root].add((input, part))
        for output, sym in zip(op.outputs, op._output_parts):
            root = output._storage.get_root()
            part = strat.get_partition(sym)
            # If we are writing to a view of a particular storage, then this
            # view should be the only view that we are reading from.
            if root in self._read_views:
                views = self._read_views[root]
                if len(views) > 1:
                    return False
                view = next(iter(views))
                if view != (output, part):
                    return False
        for reduc, _ in op.reductions:
            root = reduc._storage.get_root()
            # We cannot reduce to a store that we are currently reading from.
            # TODO (rohany): I think there's an interaction here around privilege
            #  escalation that could lead to sub-optimal fusion.
            if root in self._read_views:
                return False
        return True


# FusionConstraintManager manages a set of user-provided fusion constraints,
# and controls how much of the task buffer can be fused.
class FusionConstraintManager:
    def __init__(self):
        self._constraints = []

    def register_constraint(self, constraint: FusionConstraint) -> None:
        self._constraints.append(constraint)

    # compute_fusable_prefix returns how large of a prefix of the input
    # list of operations may be fused together.
    def compute_fusable_prefix(
        self,
        ops: List[Operation],
        strategy: List[Strategy],
        generate_new_strategy: Callable[[Operation, List[Strategy]], None]
    ) -> int:
        for idx, op in enumerate(ops):
            # If we don't have enough strategies, generate a new one.
            if idx >= len(strategy):
                generate_new_strategy(op, strategy)
                assert(idx < len(strategy))
            strat = strategy[idx]
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
        #    1b. Check that the scalar arguments to the tasks are the same.
        # 2) All input stores have same dimensions, types and transforms.
        # 3) The reference count status of each store should be the same (not the actual counts,
        #    but what was dropped versus what was held.
        # 4) The dependency graph of tasks and their store arguments need to be
        #    isomorphic. I realize now that it's not actually a graph, but a list
        #    of objects that we can determine an isomorphism. I think that we need
        #    to check for this isomorphism between both the stores and the storages
        #    as both relations are used by the fusion analysis.

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
            # TODO (rohany): This might need library deduplication.
            self.task_ids.append(op._task_id)

            # Pack task scalar arguments into comparable buffers.
            builder = BufferBuilder()
            for arg, dtype in op._scalar_args:
                ScalarArg(arg, dtype).pack(builder)
            self.task_scalar_args.append(bytes(builder.get_string()))

            op_store_inputs, op_store_outputs, op_store_reducs, op_reduc_redops = [], [], [], []
            op_storage_inputs, op_storage_outputs, op_storage_reducs = [], [], []
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
            for store, redop in op.reductions:
                oldLen = len(op_store_reducs)
                store_idctr = add_id(store._unique_id, store_idctr, store_ids, op_store_reducs)
                storage_idctr = add_id(store._storage._unique_id, storage_idctr, storage_ids, op_storage_reducs)
                if oldLen != len(op_store_reducs):
                    op_reduc_redops.append(redop)
                self.store_types.append(store.type)
                self.store_dims.append(store.ndim)
                # self.store_transforms.append(store.transform)
                self.store_liveness.append(store.has_external_references())

            # Convert the remapped store and storage lists to tuples so that
            # we can hash them later.
            self.store_generic_ids.append((tuple(op_store_inputs), tuple(op_store_outputs), tuple(op_store_reducs), tuple(op_reduc_redops)))
            self.storage_generic_ids.append((tuple(op_storage_inputs), tuple(op_storage_outputs), tuple(op_storage_reducs)))

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

    def __str__(self):
        return f"TaskWindowDescriptor(\n" \
               f"  {self.task_ids},\n" \
               f"  {self.task_scalar_args},\n" \
               f"  {self.store_types},\n" \
               f"  {self.store_dims},\n" \
               f"  {self.store_transforms},\n" \
               f"  {self.store_liveness},\n" \
               f"  {self.store_generic_ids},\n" \
               f"  {self.storage_generic_ids}\n" \
               f")"

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


# The idea here is to aggregate as much information as we need to be able
# to map from the original tasks and partitioning scheme onto the new task and stores
# for the fused task.
class FusedTaskConstructionDescriptor:
    def __init__(
            self,
            ops: List[Task],
            inputs: List[Store],
            outputs: List[Store],
            reducs: List[Store],
            local_taskid: int,
            global_taskid: int,
            funcptr: int,
    ):
        self.local_taskid = local_taskid
        self.global_taskid = global_taskid
        self.funcptr = funcptr

        # Maintain a separate mapping for each list, because if the same store
        # appears in multiple lists a single mapping would become invalid.
        store_to_orig_pos_inputs, store_to_orig_pos_outputs, store_to_orig_pos_reducs  = {}, {}, {}
        for opidx, op in enumerate(ops):
            for storeidx, store in enumerate(op.inputs):
                if store._unique_id not in store_to_orig_pos_inputs:
                    store_to_orig_pos_inputs[store._unique_id] = (opidx, storeidx)
            for storeidx, store in enumerate(op.outputs):
                if store._unique_id not in store_to_orig_pos_outputs:
                    store_to_orig_pos_outputs[store._unique_id] = (opidx, storeidx)
            for storeidx, (store, _) in enumerate(op.reductions):
                if store._unique_id not in store_to_orig_pos_reducs:
                    store_to_orig_pos_reducs[store._unique_id] = (opidx, storeidx)

        self.inputs, self.outputs, self.reducs = [], [], []
        for store in inputs:
            self.inputs.append(store_to_orig_pos_inputs[store._unique_id])
        for store in outputs:
            self.outputs.append(store_to_orig_pos_outputs[store._unique_id])
        for store in reducs:
            self.reducs.append(store_to_orig_pos_reducs[store._unique_id])

    def build_task(self, ops: List[Task], strategies: List[Strategy]) -> Tuple[Operation, Strategy]:
        # TODO (rohany): Worry about when these ops come from different libraries.
        newTask = ops[0].context.create_auto_task(self.local_taskid)
        self.add_stores_to_task(newTask, ops)
        self.add_impl_to_task(newTask)
        newStrat = self.build_new_strategy(newTask, ops, strategies)
        return newTask, newStrat

    def add_impl_to_task(self, fused: AutoTask):
        fused.add_scalar_arg(self.global_taskid, ty.uint32)

    def add_stores_to_task(self, fused: AutoTask, ops: List[Task]):
        for opidx, storeidx in self.inputs:
            op = ops[opidx]
            fused.add_input(op.inputs[storeidx])
        for opidx, storeidx in self.outputs:
            op = ops[opidx]
            fused.add_output(op.outputs[storeidx])
        for opidx, storeidx in self.reducs:
            op = ops[opidx]
            fused.add_reduction(*op.reductions[storeidx])

    # build_part_sym_mapping assumes that add_stores_to_task has already been called.
    def build_part_sym_mapping(self, fused: AutoTask, ops: List[Task]):
        part_sym_mapping = {}
        # Remap only the symbols that we are using in the fused task.
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
        # As part of building the new strategy, we promote single tasks into
        # the launch domain of everything else in the buffer. Because of the
        # launch constraint applied earlier, we know that all tasks are
        # indeed promotable.
        launch_domains = list(set(s.launch_domain for s in strategies if s.launch_domain is not None))
        # Either everything is None, or some are None and everything
        # else is the same launch domain.
        assert(len(launch_domains) == 1 or len(launch_domains) == 0)
        merged_launch_domain = launch_domains[0] if len(launch_domains) == 1 else None

        def adjust_launch_domain(strat: Strategy):
            # If we're doing a single task launch where all of the
            # launch domains are None, then no strategy modification
            # needs to be done.
            if merged_launch_domain is None:
                return strat
            if strat.launch_domain is None:
                new_strat = strat.clone()
                new_strat._launch_domain = merged_launch_domain
                return new_strat
            return strat

        # Merge all of the individual strategies into a single strategy.
        new_strat = adjust_launch_domain(strategies[0])
        for strat in strategies[1:]:
            new_strat.merge(adjust_launch_domain(strat))
        # Remap all of the needed symbols over to the new symbols.
        return new_strat.remap(self.build_part_sym_mapping(fused, ops))


# Utility methods to perform components of task and kernel fusion.
def generate_mlir_modules(ops: List[Operation]) -> List[PyMLIRModule]:
    from .launcher import ScalarArg

    modules = []
    for op in ops:
        tid = op._task_id
        info = op.context._cpp_context.find_task(tid)
        gen = info.get_mlir_body_generator()
        inputs = [s.to_comp_time_store_desc() for s in op.inputs]
        outputs = [s.to_comp_time_store_desc() for s in op.outputs]
        reducs = [s[0].to_comp_time_store_desc() for s in op.reductions]

        # Pack the arguments into a buffer for the tasks.
        builder = BufferBuilder()
        for arg, dtype in op._scalar_args:
            scalar = ScalarArg(arg, dtype)
            scalar.pack(builder)
        buffer = bytes(builder.get_string())
        bufSize = builder.get_size()
        # TODO (rohany): Include a cache for these generated kernels.
        modules.append(gen.generate_body(inputs, outputs, reducs, buffer, bufSize))
    return modules


# The calling convention for fused tasks is to deduplicate and group
# the inputs, outputs and reductions for the new task.
def construct_store_calling_convention(ops: List[Operation]) -> Tuple[List[Store], List[Store], List[Store]]:
    new_inputs, new_outputs, new_reducs = [], [], []
    dedup_inputs, dedup_outputs, dedup_reducs = set(), set(), set()
    for op in ops:
        def dedup_add(l, s, val):
            if val not in s:
                s.add(val)
                l.append(val)
        for input in op.inputs:
            dedup_add(new_inputs, dedup_inputs, input)
        for output in op.outputs:
            dedup_add(new_outputs, dedup_outputs, output)
        for reduc, _ in op.reductions:
            dedup_add(new_reducs, dedup_reducs, reduc)
    return new_inputs, new_outputs, new_reducs
