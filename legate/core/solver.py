# Copyright 2021-2022 NVIDIA Corporation
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
from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Generic, List, Optional, Tuple, TypeVar

from . import FieldSpace, Future, Rect
from .constraints import Alignment, Broadcast, Containment, PartSym
from .partition import REPLICATE
from .runtime import runtime
from .shape import Shape
from .utils import OrderedSet

if TYPE_CHECKING:
    from .constraints import Expr, Lit
    from .operation import Operation
    from .partition import PartitionBase
    from .store import Store
    from .transform import Restrictions


def join_restrictions(x: Restrictions, y: Restrictions) -> Restrictions:
    return tuple(min(a, b) for a, b in zip(x, y))


T = TypeVar("T")


class EqClass(Generic[T]):
    def __init__(self) -> None:
        # Maps a variable to the equivalent class id
        self._class_ids: dict[T, int] = {}
        self._next_class_id = 0
        # Maps an equivalent class id to the class
        self._classes: dict[int, OrderedSet[T]] = {}

    @property
    def empty(self) -> bool:
        return self._next_class_id == 0

    def _add(self, var1: T, var2: T) -> None:
        cls = OrderedSet([var1, var2])
        cls_id = self._next_class_id
        self._next_class_id += 1
        self._classes[cls_id] = cls
        self._class_ids[var1] = cls_id
        self._class_ids[var2] = cls_id

    def _update(self, var1: T, var2: T) -> None:
        cls_id = self._class_ids[var1]
        cls = self._classes[cls_id]
        cls.add(var2)
        self._class_ids[var2] = cls_id

    def _merge(self, var1: T, var2: T) -> None:
        cls_id1 = self._class_ids[var1]
        cls_id2 = self._class_ids[var2]
        new_cls: OrderedSet[T] = OrderedSet()
        new_cls.update(self._classes[cls_id1])
        new_cls.update(self._classes[cls_id2])
        self._classes[cls_id1] = new_cls
        self._classes[cls_id2] = new_cls

    # TODO (rohany): Comment...
    def remap(self, mapping: dict[T, T]) -> EqClass[T]:
        def map(val: T) -> T:
            if val in mapping:
                return mapping[val]
            return val
        new_classes: dict[int, OrderedSet[T]] = {}
        for k, s in self._classes.items():
            new_set: OrderedSet[T] = OrderedSet()
            for val in s:
                new_set.add(map(val))
            new_classes[map(k)] = s
        result = EqClass()
        result._class_ids = {map(k): v for k, v in self._class_ids.items()}
        result._next_class_id = self._next_class_id
        result._classes = new_classes
        return result


    def record(self, var1: T, var2: T) -> None:
        """
        Record an equivalence relation between two vars
        """
        found1 = var1 in self._class_ids
        found2 = var2 in self._class_ids

        if found1 and found2:  # both
            self._merge(var1, var2)
        elif found1:  # first only
            self._update(var1, var2)
        elif found2:  # second only
            self._update(var2, var1)
        else:  # neither
            self._add(var1, var2)

    def copy(self) -> EqClass[T]:
        new: EqClass[T] = EqClass()
        new._class_ids = self._class_ids.copy()
        new._classes = self._classes.copy()
        return new

    def union(self, other: EqClass[T]) -> None:
        if self.empty:
            self._class_ids = other._class_ids.copy()
            self._classes = other._classes.copy()
        else:
            for other_class in other._classes.values():
                cls: OrderedSet[T] = OrderedSet()
                cls.update(other_class)
                var1 = cls.pop()
                for var2 in cls:
                    self.record(var1, var2)

    def find(self, var: T) -> OrderedSet[T]:
        """
        Return an equivalence class for a given var.
        """
        if var not in self._class_ids:
            return OrderedSet([var])
        else:
            return self._classes[self._class_ids[var]]

    def aligned(self, var1: T, var2: T) -> bool:
        return self._class_ids[var1] == self._class_ids[var2]


class Strategy:
    _launch_domain: Optional[Rect]

    def __init__(
        self,
        launch_shape: Optional[Shape],
        strategy: dict[PartSym, PartitionBase],
        fspaces: dict[PartSym, FieldSpace],
        key_parts: set[PartSym],
        eq_classes: EqClass[PartSym],
    ) -> None:
        if launch_shape is not None:
            self._launch_domain = Rect(hi=launch_shape)
        else:
            self._launch_domain = None
        self._strategy = strategy
        self._fspaces = fspaces
        self._key_parts = key_parts
        self._eq_classes = eq_classes

    @property
    def parallel(self) -> bool:
        return self._launch_domain is not None

    @property
    def launch_domain(self) -> Optional[Rect]:
        return self._launch_domain

    @property
    def launch_ndim(self) -> int:
        if self._launch_domain is None:
            return 1
        else:
            return self._launch_domain.dim

    def aligned(self, symb1: PartSym, symb2: PartSym) -> bool:
        return self._eq_classes.aligned(symb1, symb2)

    def set_launch_domain(self, launch_domain: Rect) -> None:
        if self._launch_domain is not None:
            raise RuntimeError(
                "Manual task launch cannot be used when some stores are "
                "auto-partitioned"
            )
        self._launch_domain = launch_domain

    def get_partition(self, part: PartSym) -> PartitionBase:
        assert not part.store.unbound
        if part not in self._strategy:
            raise ValueError(f"No strategy is found for {part}")
        return self._strategy[part]

    def get_field_space(self, part: PartSym) -> FieldSpace:
        assert part.store.unbound
        if part not in self._fspaces:
            raise ValueError(f"No strategy is found for {part}")
        return self._fspaces[part]

    def is_key_part(self, part: PartSym) -> bool:
        return part in self._key_parts

    def unify_key_part(self, part1: PartSym, part2: PartSym) -> None:
        non_members = {part1, part2} - self._key_parts
        if len(non_members) == 1:
            self._key_parts.update(non_members)

    # Remap a strategy to a new set of partition symbols.
    # TODO (rohany): I'm unsure if we'll actually need this. We might
    #  need a "merge" function that takes in multiple strategies.
    def remap(self, mapping: dict[PartSym, PartSym]) -> Strategy:
        def map(sym: PartSym) -> PartSym:
            if sym in mapping:
                return mapping[sym]
            return sym
        new_strategy = {map(sym): part for sym, part in self._strategy.items()}
        new_fspaces = {map(sym): fspace for sym, fspace in self._fspaces.items()}
        new_key_parts = {map(sym) for sym in self._key_parts}
        new_eq_classes = self._eq_classes.remap(mapping)
        new_launch_domain = (Shape(self._launch_domain.hi) + 1) if self._launch_domain is not None else None
        return Strategy(new_launch_domain, new_strategy, new_fspaces, new_key_parts, new_eq_classes)

    # Merge this strategy with another partitioning strategy, destructively
    # updating the current strategy.
    def merge(self, other: Strategy):
        assert(self._launch_domain == other._launch_domain)
        for sym, part in other._strategy.items():
            # TODO (rohany): Are repeats allowed? What happens here?
            assert(sym not in self._strategy)
            self._strategy[sym] = part
        for sym, fspace in other._fspaces.items():
            assert(sym not in self._fspaces)
            self._fspaces[sym] = fspace
        for sym in other._key_parts:
            assert(sym not in self._key_parts)
            self._key_parts.add(sym)
        self._eq_classes.union(other._eq_classes)

    def clone(self):
        return copy.copy(self)

    def __str__(self) -> str:
        st = "[Strategy]"
        st += f"\nLaunch domain: {self._launch_domain}"
        for part, partition in self._strategy.items():
            st += f"\n{part} ~~> {partition}"
        for part, fspace in self._fspaces.items():
            st += f"\n{part} ~~> {fspace}"
        return st

    def __repr__(self) -> str:
        return str(self)


class Partitioner:
    def __init__(
        self,
        ops: List[Operation],
        must_be_single: bool = False,
    ):
        self._ops = ops
        self._must_be_single = must_be_single

    def _solve_constraints_for_futures(
        self,
        unknowns: OrderedSet[PartSym],
        constraints: EqClass[PartSym],
        partitions: dict[PartSym, PartitionBase],
    ) -> OrderedSet[PartSym]:
        to_remove: OrderedSet[PartSym] = OrderedSet()
        for unknown in unknowns:
            store = unknown.store
            if store.kind is not Future:
                continue

            to_remove.add(unknown)

            if unknown in partitions:
                continue

            if store.kind is Future:
                partitions[unknown] = REPLICATE
            else:
                cls = constraints.find(unknown)
                for to_align in cls:
                    partitions[to_align] = REPLICATE

        return unknowns.remove_all(to_remove)

    def _solve_unbound_constraints(
        self,
        unknowns: OrderedSet[PartSym],
        constraints: EqClass[PartSym],
        partitions: dict[PartSym, PartitionBase],
        fspaces: dict[PartSym, FieldSpace],
    ) -> tuple[OrderedSet[PartSym], Optional[int]]:
        to_remove: OrderedSet[PartSym] = OrderedSet()
        for unknown in unknowns:
            store = unknown.store
            if not store.unbound:
                continue

            to_remove.add(unknown)

            if unknown in partitions:
                continue

            cls = constraints.find(unknown)
            assert all(to_align.store.unbound for to_align in cls)

            fspace = runtime.create_field_space()
            for to_align in cls:
                partitions[to_align] = REPLICATE
                fspaces[to_align] = fspace

        unbound_ndims = set(unknown.store.ndim for unknown in to_remove)

        if len(unbound_ndims) > 1:
            raise NotImplementedError(
                "Unbound stores for an operation must have the same "
                "number of dimensions for now"
            )

        unbound_ndim = unbound_ndims.pop() if len(unbound_ndims) == 1 else None
        return unknowns.remove_all(to_remove), unbound_ndim

    @staticmethod
    def _find_restrictions(
        cls: OrderedSet[PartSym], broadcasts: dict[PartSym, Restrictions]
    ) -> Restrictions:
        merged: Optional[Restrictions] = None
        for unknown in cls:
            store = unknown.store
            restrictions = store.find_restrictions()
            if unknown in broadcasts:
                restrictions = join_restrictions(
                    broadcasts[unknown], restrictions
                )
            if merged is None:
                merged = restrictions
            else:
                merged = join_restrictions(merged, restrictions)
        assert merged is not None
        return merged

    def _find_all_restrictions(
        self,
        unknowns: OrderedSet[PartSym],
        broadcasts: dict[PartSym, Restrictions],
        constraints: EqClass[PartSym],
    ) -> dict[PartSym, Restrictions]:
        all_restrictions: dict[PartSym, Restrictions] = {}
        for unknown in unknowns:
            if unknown in all_restrictions:
                continue
            cls = constraints.find(unknown)
            restrictions = self._find_restrictions(cls, broadcasts)
            for store in cls:
                all_restrictions[unknown] = restrictions
        return all_restrictions

    def maybe_find_alternative_key_partition(
        self,
        chosen_partition: PartitionBase,
        original: PartSym,
        cls: OrderedSet[PartSym],
        restrictions: Restrictions,
        must_be_even: OrderedSet[PartSym],
    ) -> tuple[PartitionBase, PartSym]:
        original_comm_vol = original.store.comm_volume()
        # If there is a store in the equivalence class that has an even
        # key partition, we use it instead
        for unknown in cls:
            store = unknown.store
            # Careful! these are partition symbols and we overrode the equal
            # operator on them to mean alignments, so we compare between
            # their identities.
            if original is unknown:
                continue
            elif not store.has_key_partition(restrictions):
                continue
            # We don't want to use the store's key partition if that store
            # incurs a bigger amount of data movement
            elif store.comm_volume() < original_comm_vol:
                continue

            part = store.compute_key_partition(restrictions)
            if part.even:
                return part, unknown

        # TODO: For now we repartition the store used as a center of a stencil
        # if it was previously partitioned unevenly. If we have a proper
        # affine dependent partitioning calls, we can avoid this.
        if original in must_be_even:
            store = original.store
            store.reset_key_partition()
            return store.compute_key_partition(restrictions), original
        else:
            return chosen_partition, original

    @staticmethod
    def compute_launch_shape(
        partitions: dict[PartSym, PartitionBase],
        all_outputs: set[Store],
        unbound_ndim: Optional[int],
        # The Boolean return value denotes whether the computed launch shape
        # is "final". If it's True, there's no room for the solver to improve
        # the quality of parallelization.
    ) -> Tuple[Optional[Shape], bool]:
        # We filter out the cases where any of the outputs is assigned
        # to replication, in which case the operation must be performed
        # sequentially
        for unknown, part in partitions.items():
            if unknown.store in all_outputs and part is REPLICATE:
                return None, True

        # If we're here, this means that replicated stores are safe to access
        # in parallel, so we filter those out to determine the launch domain
        parts = [part for part in partitions.values() if part is not REPLICATE]

        # If all stores are replicated, we can't parallelize the operation
        if len(parts) == 0:
            return None, True

        # Here we check if all partitions agree on the color shape
        must_be_1d_launch = False
        launch_shape = parts[0].color_shape
        assert launch_shape is not None
        for part in parts[1:]:
            if part.color_shape != launch_shape:
                # When some partitions have different color shapes,
                # a 1D launch space is the only option
                must_be_1d_launch = True
                break

        if must_be_1d_launch:
            # If this operation has a multi-dimensional unbound store,
            # we can't use a 1-D launch domain, hence falling back to
            # a sequential launch
            if unbound_ndim is not None and unbound_ndim != 1:
                return None, True

            # If all color spaces don't have the same number of colors,
            # it means some inputs are much smaller than the others
            # to be partitioned into the same number of pieces.
            # We simply serialize the launch in that case for now.
            volumes = set()
            for part in parts:
                assert part.color_shape is not None
                volumes.add(part.color_shape.volume())
            if len(volumes) > 1:
                return None, False
            else:
                return Shape(volumes), True
        # If there is an unbound store, the store's dimensionality must be
        # the same as that of the launch domain
        elif unbound_ndim is None or unbound_ndim == launch_shape.ndim:
            return launch_shape, True
        else:
            return None, True

    def _solve_store_constraints(
        self,
        partitions: dict[PartSym, PartitionBase],
        unknowns: list[PartSym],
        dependent: dict[PartSym, Expr],
        all_restrictions: dict[PartSym, Restrictions],
        constraints: EqClass[PartSym],
        must_be_even: OrderedSet[PartSym],
    ) -> Tuple[dict[PartSym, PartitionBase], set[PartSym]]:
        result = partitions.copy()
        key_parts: set[PartSym] = set()

        for unknown in unknowns:
            if unknown in result:
                continue
            elif unknown in dependent:
                continue

            store = unknown.store
            restrictions = all_restrictions[unknown]
            cls = constraints.find(unknown)

            partition = store.compute_key_partition(restrictions)
            if not partition.even and len(cls) > 1:
                partition, unknown = self.maybe_find_alternative_key_partition(
                    partition,
                    unknown,
                    cls,
                    restrictions,
                    must_be_even,
                )
            key_parts.add(unknown)

            for to_align in cls:
                if to_align in result:
                    continue
                result[to_align] = partition

        for rhs, lhs in dependent.items():
            expr = lhs.subst(result).reduce()
            if TYPE_CHECKING:
                assert isinstance(expr, Lit)
            result[rhs] = expr._part

        return result, key_parts

    @staticmethod
    def _reset_less_optimal_partitions(
        partitions: dict[PartSym, PartitionBase],
    ) -> bool:
        valid_parts = [
            (part.color_shape.volume(), unknown)
            for unknown, part in partitions.items()
            if part.color_shape is not None
        ]
        max_dop = max(dop for dop, _ in valid_parts)
        reset_any = False
        for dop, unknown in valid_parts:
            if dop == max_dop:
                continue
            reset_any = True
            unknown.store.reset_key_partition()

        return reset_any

    def partition_stores(self) -> Strategy:
        unknowns: OrderedSet[PartSym] = OrderedSet()
        constraints: EqClass[PartSym] = EqClass()
        broadcasts: dict[PartSym, Restrictions] = {}
        dependent: dict[PartSym, Expr] = {}
        must_be_even: OrderedSet[PartSym] = OrderedSet()
        all_outputs: set[Store] = set()
        for op in self._ops:
            unknowns.update(op.all_unknowns)
            for c in op.constraints:
                if isinstance(c, Alignment):
                    constraints.record(c._lhs, c._rhs)
                elif isinstance(c, Broadcast):
                    broadcasts[c._expr] = c._restrictions
                elif isinstance(c, Containment) and isinstance(
                    c._lhs, PartSym
                ):
                    if c._lhs in dependent:
                        raise NotImplementedError(
                            "Partitions constrained by multiple constraints "
                            "are not supported yet"
                        )
                    for unknown in c._rhs.unknowns():
                        must_be_even.add(unknown)
                    dependent[c._lhs] = c._rhs
                elif isinstance(c, Containment) and isinstance(
                    c._rhs, PartSym
                ):
                    if c._rhs in dependent:
                        raise NotImplementedError(
                            "Partitions constrained by multiple constraints "
                            "are not supported yet"
                        )
                    for unknown in c._lhs.unknowns():
                        must_be_even.add(unknown)
                    dependent[c._rhs] = c._lhs
        for op in self._ops:
            all_outputs.update(
                store for store in op.outputs if not store.unbound
            )

        if self._must_be_single or len(unknowns) == 0:
            for unknown in unknowns:
                c = unknown.broadcast()
                broadcasts[unknown] = c._restrictions

        partitions: dict[PartSym, PartitionBase] = {}
        fspaces: dict[PartSym, FieldSpace] = {}

        unknowns = self._solve_constraints_for_futures(
            unknowns,
            constraints,
            partitions,
        )

        unknowns, unbound_ndim = self._solve_unbound_constraints(
            unknowns,
            constraints,
            partitions,
            fspaces,
        )

        all_restrictions = self._find_all_restrictions(
            unknowns, broadcasts, constraints
        )

        def cost(unknown: PartSym) -> tuple[int, bool]:
            store = unknown.store
            return (
                -store.comm_volume(),
                not store.has_key_partition(all_restrictions[unknown]),
                # This is a temporary hack to help with stencil codes, where we
                # prefer the key partitions of temporary stores. This helps to
                # keep temporary stores in place and move some of the input data,
                # which is actually the right choice for stencil codes.
                store.has_external_references(),
            )

        result: dict[PartSym, PartitionBase]
        key_parts: set[PartSym]
        launch_shape: Optional[Shape]

        can_retry = True
        while True:
            result, key_parts = self._solve_store_constraints(
                partitions,
                sorted(unknowns, key=cost),
                dependent,
                all_restrictions,
                constraints,
                must_be_even,
            )

            launch_shape, done = self.compute_launch_shape(
                result, all_outputs, unbound_ndim
            )
            # When partitions have different numbers of chunks, the solver
            # normally decides to serialize the operation, as there's no
            # obvious mapping between the partitions. However, it is
            # sometimes possible to recover parallelism by searching for
            # an alternative solution to the given set of partitioning
            # constraints, especially when some of the stores have cached key
            # partitions that are not computed for themselves but copied from
            # others due to alignments.
            if can_retry and not done:
                # We only retry once because resetting the cached key
                # partitions followed recomputing key partitions is
                # idempotent.
                can_retry = False
                if self._reset_less_optimal_partitions(result):
                    continue
            break

        return Strategy(launch_shape, result, fspaces, key_parts, constraints)
