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

import weakref
from typing import TYPE_CHECKING, Any, Optional, Sequence, Type, Union

from . import (
    Attach,
    Detach,
    Future,
    IndexAttach,
    IndexDetach,
    InlineMapping,
    Point,
    ffi,
    legion,
)
from .allocation import (
    Attachable,
    DistributedAllocation,
    InlineMappedAllocation,
)
from .partition import REPLICATE, PartitionBase, Restriction, Tiling, Weighted
from .projection import execute_functor_symbolically
from .runtime import runtime
from .shape import Shape
from .transform import (
    Delinearize,
    Project,
    Promote,
    Shift,
    TransformStack,
    Transpose,
    identity,
)
from .types import _Dtype

if TYPE_CHECKING:
    from . import (
        AffineTransform,
        BufferBuilder,
        Partition as LegionPartition,
        PhysicalRegion,
        Rect,
        Region,
    )
    from .context import Context
    from .launcher import Proj
    from .projection import ProjFn
    from .transform import Transform, TransformStackBase

from math import prod

attachment_manager = runtime.attachment_manager
partition_manager = runtime.partition_manager


# A Field holds a reference to a field in a region tree
class Field:
    def __init__(
        self,
        region: Region,
        field_id: int,
        field_size: int,
        shape: Shape,
    ) -> None:
        self.region = region
        self.field_id = field_id
        self.field_size = field_size
        self.shape = shape

    def same_handle(self, other: Field) -> bool:
        return type(self) == type(other) and self.field_id == other.field_id

    def __str__(self) -> str:
        return f"Field({self.field_id})"

    def __del__(self) -> None:
        # Return our field back to the runtime
        runtime.free_field(
            self.region,
            self.field_id,
            self.field_size,
            self.shape,
        )


# A region field holds a reference to a field in a logical region
class RegionField:
    def __init__(
        self,
        region: Region,
        field: Field,
        shape: Shape,
        parent: Optional[RegionField] = None,
    ) -> None:
        self.region = region
        # Note that self.region can be different from self.field.region
        # if this is for a subregion
        self.field = field
        self.shape = shape
        self.parent = parent
        # External allocation we attached to this field
        self.attached_alloc: Union[None, Attachable] = None
        self.detach_key: int = -1
        # Physical region for attach
        self.physical_region: Union[None, PhysicalRegion] = None
        self.physical_region_refs = 0
        self.physical_region_mapped = False

        self._partitions: dict[Tiling, LegionPartition] = {}

    def __del__(self) -> None:
        if self.attached_alloc is not None:
            self.detach_external_allocation(unordered=True, defer=True)

    @staticmethod
    def create(
        region: Region, field_id: int, field_size: int, shape: Shape
    ) -> RegionField:
        field = Field(region, field_id, field_size, shape)
        return RegionField(region, field, shape)

    def same_handle(self, other: RegionField) -> bool:
        return type(self) == type(other) and self.field.same_handle(
            other.field
        )

    def __str__(self) -> str:
        return (
            f"RegionField("
            f"tid: {self.region.handle.tree_id}, "
            f"is: {self.region.handle.index_space.id}, "
            f"fs: {self.region.handle.field_space.id}, "
            f"fid: {self.field.field_id})"
        )

    def attach_external_allocation(
        self, context: Context, alloc: Attachable, share: bool
    ) -> None:
        assert self.parent is None
        # If we already have some memory attached, detach it first
        if self.attached_alloc is not None:
            raise RuntimeError("A RegionField cannot be re-attached")
        # All inline mappings should have been unmapped by now
        assert self.physical_region_refs == 0
        # Record the attached memory ranges, and confirm no overlaps with
        # previously encountered ranges.
        attachment_manager.attach_external_allocation(alloc, self)

        def record_detach(detach: Union[Detach, IndexDetach]) -> None:
            # Dangle these fields off the detachment operation, to prevent
            # premature collection
            detach.field = self.field  # type: ignore[union-attr]
            detach.alloc = alloc  # type: ignore[union-attr]
            # Don't store the detachment operation here, instead register it
            # on the attachment manager and record its unique key
            # TODO: This might not be necessary anymore
            self.detach_key = attachment_manager.register_detachment(detach)

        if isinstance(alloc, memoryview):
            # Singleton attachment
            attach = Attach(
                self.region,
                self.field.field_id,
                alloc,
                mapper=context.mapper_id,
            )
            # If we're not sharing then there is no need to map or restrict the
            # attachment
            if not share:
                attach.set_restricted(False)
                attach.set_mapped(False)
            else:
                self.physical_region_mapped = True
            # Singleton allocations return a physical region for the entire
            # domain, that can be inline-mapped directly.
            self.physical_region = runtime.dispatch(attach)
            # Add a reference here to prevent collection in inline mapped
            # cases. This reference will never be removed, we'll delete the
            # physical region once the object is deleted.
            self.physical_region_refs = 1
            # Due to the working of the Python interpreter's garbage collection
            # algorithm we make the detach operation for this now and register
            # it with the runtime so that we know that it won't be collected
            # when the RegionField object is collected.
            # We don't need to flush the contents back to the attached memory
            # if this is an internal temporary allocation.
            record_detach(Detach(self.physical_region, flush=share))
        else:
            # Distributed attachment
            assert alloc.partition.parent is self.region
            field_type = self.region.field_space.get_type(self.field.field_id)
            field_size = (
                field_type.size
                if isinstance(field_type, _Dtype)
                else field_type
            )
            shard_local_data = {}
            for (c, buf) in alloc.shard_local_buffers.items():
                subregion = alloc.partition.get_child(c)
                bounds = subregion.index_space.get_bounds()
                if buf.shape != tuple(
                    bounds.hi[i] - bounds.lo[i] + 1 for i in range(bounds.dim)
                ):
                    raise RuntimeError(
                        "Subregion shape does not match attached buffer"
                    )
                if buf.itemsize != field_size:
                    raise RuntimeError(
                        "Field type does not match attached buffer"
                    )
                shard_local_data[subregion] = buf
            index_attach = IndexAttach(
                self.region,
                self.field.field_id,
                shard_local_data,
                mapper=context.mapper_id,
            )
            index_attach.set_deduplicate_across_shards(True)
            # If we're not sharing there is no need to restrict the attachment
            if not share:
                index_attach.set_restricted(False)
            external_resources = runtime.dispatch(index_attach)
            # We don't need to flush the contents back to the attached memory
            # if this is an internal temporary allocation.
            record_detach(IndexDetach(external_resources, flush=share))
        # Record the attachment
        self.attached_alloc = alloc

    def detach_external_allocation(
        self, unordered: bool, defer: bool = False
    ) -> None:
        assert self.parent is None
        assert self.attached_alloc is not None
        detach = attachment_manager.remove_detachment(self.detach_key)
        detach.unordered = unordered  # type: ignore[union-attr]
        attachment_manager.detach_external_allocation(
            self.attached_alloc, detach, defer
        )
        self.physical_region = None
        self.physical_region_mapped = False
        self.physical_region_refs = 0
        self.attached_alloc = None

    def get_inline_mapped_region(self, context: Context) -> PhysicalRegion:
        if self.parent is None:
            if self.physical_region is None:
                # We don't have a valid numpy array so we need to do an inline
                # mapping and then use the buffer to share the storage
                mapping = InlineMapping(
                    self.region,
                    self.field.field_id,
                    mapper=context.mapper_id,
                )
                self.physical_region = runtime.dispatch(mapping)
                self.physical_region_mapped = True
                # Wait until it is valid before returning
                self.physical_region.wait_until_valid()
            elif not self.physical_region_mapped:
                # If we have a physical region but it is not mapped then
                # we actually need to remap it, we do this by launching it
                runtime.dispatch(self.physical_region)
                self.physical_region_mapped = True
                # Wait until it is valid before returning
                self.physical_region.wait_until_valid()
            # Increment our ref count so we know when it can be collected
            self.physical_region_refs += 1
            return self.physical_region
        else:
            return self.parent.get_inline_mapped_region(context)

    def decrement_inline_mapped_ref_count(
        self, unordered: bool = False
    ) -> None:
        if self.parent is None:
            if self.physical_region is None:
                return
            assert self.physical_region_refs > 0
            self.physical_region_refs -= 1
            if self.physical_region_refs == 0:
                runtime.unmap_region(self.physical_region, unordered=unordered)
                self.physical_region = None
                self.physical_region_mapped = False
        else:
            self.parent.decrement_inline_mapped_ref_count(unordered=unordered)

    def get_inline_allocation(
        self,
        shape: Shape,
        context: Optional[Context] = None,
        transform: Optional[AffineTransform] = None,
    ) -> InlineMappedAllocation:
        context = runtime.core_context if context is None else context

        physical_region = self.get_inline_mapped_region(context)
        # We need a pointer to the physical allocation for this physical region
        dim = max(shape.ndim, 1)
        # Build the accessor for this physical region
        if transform is not None:
            # We have a transform so build the accessor special with a
            # transform
            func = getattr(
                legion,
                "legion_physical_region_get_field_accessor_array_"
                f"{dim}d_with_transform",
            )
            accessor = func(
                physical_region.handle,
                ffi.cast("legion_field_id_t", self.field.field_id),
                transform.raw(),
            )
        else:
            # No transfrom so we can do the normal thing
            func = getattr(
                legion,
                f"legion_physical_region_get_field_accessor_array_{dim}d",
            )
            accessor = func(
                physical_region.handle,
                ffi.cast("legion_field_id_t", self.field.field_id),
            )
        # Now that we've got our accessor we can get a pointer to the memory
        rect = ffi.new(f"legion_rect_{dim}d_t *")
        if shape.ndim == 0:
            rect[0].lo.x[0] = 0
            rect[0].hi.x[0] = 0  # inclusive
        else:
            for d in range(dim):
                rect[0].lo.x[d] = 0
                rect[0].hi.x[d] = shape[d] - 1  # inclusive
        subrect = ffi.new(f"legion_rect_{dim}d_t *")
        offsets = ffi.new("legion_byte_offset_t[]", dim)
        func = getattr(legion, f"legion_accessor_array_{dim}d_raw_rect_ptr")
        base_ptr = func(accessor, rect[0], subrect, offsets)
        assert base_ptr is not None
        # Check that the subrect is the same as in the in rect
        for d in range(dim):
            assert rect[0].lo.x[d] == subrect[0].lo.x[d]
            assert rect[0].hi.x[d] == subrect[0].hi.x[d]
        strides = tuple(offsets[i].offset for i in range(dim))
        # Numpy doesn't know about CFFI pointers, so we have to cast
        # this to a Python long before we can hand it off to Numpy.
        ptr = ffi.cast("size_t", base_ptr)
        return InlineMappedAllocation(
            self,
            tuple(shape) if shape.ndim > 0 else (1,),
            int(ptr),  # type: ignore[call-overload]
            strides,
        )

    def register_consumer(self, consumer: Any) -> None:
        # We add a callback that will be triggered when the consumer object is
        # collected. This callback carries a (captured) reference to the source
        # RegionField, keeping it alive while any consumers remain. Note that
        # weakref.ref() would not work for this purpose, because callbacks
        # passed to weakref.ref() do NOT keep their pointed objects alive. We
        # avoid storing references from the source RegionField to the consumer,
        # so that we don't create reference cycles.

        def callback() -> None:
            self.decrement_inline_mapped_ref_count()

        weakref.finalize(consumer, callback)

    def get_child(
        self, functor: Tiling, color: Shape, complete: bool = False
    ) -> RegionField:
        if functor in self._partitions:
            partition = self._partitions[functor]
        else:
            part = functor.construct(self.region, complete=complete)
            assert part is not None
            partition = part
            self._partitions[functor] = partition

        child_region = partition.get_child(Point(color))
        return RegionField(
            child_region,
            self.field,
            functor.get_subregion_size(self.shape, color),
            self,
        )


class StoragePartition:
    def __init__(
        self,
        level: int,
        parent: Storage,
        partition: PartitionBase,
        complete: bool = False,
    ) -> None:
        self._level = level
        self._parent = parent
        self._partition = partition
        self._complete = complete
        self._child_data: dict[Shape, RegionField] = {}
        self._child_sizes: dict[Shape, Shape] = {}
        self._child_offsets: dict[Shape, Shape] = {}

    @property
    def parent(self) -> Storage:
        return self._parent

    @property
    def level(self) -> int:
        return self._level

    def get_root(self) -> Storage:
        return self._parent.get_root()

    def get_child(self, color: Shape) -> Storage:
        assert isinstance(self._partition, Tiling)
        if not self._partition.has_color(color):
            raise ValueError(
                f"{color} is not a valid color for {self._partition}"
            )
        extents = self.get_child_size(color)
        offsets = self.get_child_offsets(color)
        return Storage(
            extents,
            self._level + 1,
            self._parent.dtype,
            parent=self,
            color=color,
            offsets=offsets,
        )

    def get_child_size(self, color: Shape) -> Shape:
        assert isinstance(self._partition, Tiling)
        if color not in self._child_sizes:
            assert isinstance(self._partition, Tiling)
            assert self._parent.extents is not None
            size = self._partition.get_subregion_size(
                self._parent.extents, color
            )
            self._child_sizes[color] = size
            return size
        else:
            return self._child_sizes[color]

    def get_child_offsets(self, color: Shape) -> Shape:
        assert isinstance(self._partition, Tiling)
        if color not in self._child_offsets:
            offsets = self._partition.get_subregion_offsets(color)
            self._child_offsets[color] = offsets
            return offsets
        else:
            return self._child_offsets[color]

    def get_child_data(self, color: Shape) -> RegionField:
        assert isinstance(self._partition, Tiling)
        if color not in self._child_data:
            parent_data = self._parent.data
            assert isinstance(self._partition, Tiling)
            assert isinstance(parent_data, RegionField)
            data = parent_data.get_child(
                self._partition, color, self._complete
            )
            self._child_data[color] = data
            return data
        else:
            return self._child_data[color]

    def find_key_partition(
        self, restrictions: tuple[Restriction, ...]
    ) -> Optional[PartitionBase]:
        return self._parent.find_key_partition(restrictions)

    def find_or_create_legion_partition(self) -> Optional[LegionPartition]:
        return self._parent.find_or_create_legion_partition(
            self._partition, self._complete
        )

    def is_disjoint_for(self, launch_domain: Optional[Rect]) -> bool:
        return self._partition.is_disjoint_for(launch_domain)


class Storage:
    def __init__(
        self,
        extents: Optional[Shape],
        level: int,
        dtype: Any,
        data: Optional[Union[RegionField, Future]] = None,
        kind: type = RegionField,
        parent: Optional[StoragePartition] = None,
        color: Optional[Shape] = None,
        offsets: Optional[Shape] = None,
    ) -> None:
        assert (
            data is None
            or isinstance(data, RegionField)
            or isinstance(data, Future)
        )
        assert not isinstance(data, Future) or parent is None
        assert parent is None or color is not None
        self._extents = extents
        self._offsets = offsets
        self._level = level
        self._dtype = dtype
        self._data = data
        self._kind = kind
        self._parent = parent
        self._color = color
        self._partitions: dict[PartitionBase, Optional[LegionPartition]] = {}
        self._key_partition: Union[None, PartitionBase] = None

        if self._offsets is None and self._extents is not None:
            self._offsets = Shape((0,) * self._extents.ndim)

        # True means this storage is transferrable
        self._linear = False
        # True means this storage is transferred
        self._transferred = False

    def __str__(self) -> str:
        return (
            f"{self._kind.__name__}(uninitialized)"
            if self._data is None
            else str(self._data)
        )

    def __repr__(self) -> str:
        return str(self)

    @property
    def extents(self) -> Shape:
        if self._extents is None:
            runtime.flush_scheduling_window()
            if self._extents is None:
                raise ValueError(
                    "Illegal to access an uninitialized unbound store"
                )
        return self._extents

    @property
    def offsets(self) -> Shape:
        if self._offsets is None:
            self._offsets = Shape((0,) * self.extents.ndim)
        return self._offsets

    @property
    def kind(self) -> type:
        return self._kind

    @property
    def dtype(self) -> Any:
        return self._dtype

    @property
    def ndim(self) -> int:
        return self.extents.ndim

    @property
    def data(self) -> Union[RegionField, Future]:
        """
        Return the Legion container of this Storage, which is of
        the type `self.kind`.
        """
        # If someone is trying to retreive the storage of a store,
        # we need to execute outstanding operations so that we know
        # it has been initialized correctly.
        if self._transferred:
            raise ValueError(
                "Storage is already transferred. Reusing a linear store "
                "is illegal."
            )
        runtime.flush_scheduling_window()
        if self._data is None:
            if self._kind is Future:
                raise ValueError("Illegal to access an uninitialized storage")
            if self._parent is None:
                self._data = runtime.allocate_field(self.extents, self._dtype)
            else:
                assert self._color
                self._data = self._parent.get_child_data(self._color)

        return self._data

    @property
    def has_data(self) -> bool:
        return self._data is not None

    def set_data(self, data: Union[RegionField, Future]) -> None:
        assert (
            self._kind is Future and type(data) is Future
        ) or self._data is None
        self._data = data

    @property
    def linear(self) -> bool:
        return self._linear

    def set_linear(self) -> None:
        self._linear = True

    def move_data(self, other: Storage) -> None:
        assert other._linear
        assert other.has_data
        assert not self.has_data
        other._transferred = True
        self._data = other._data
        other._data = None

    def set_extents(self, extents: Shape) -> None:
        self._extents = extents
        self._offsets = Shape((0,) * extents.ndim)

    @property
    def has_parent(self) -> bool:
        return self._parent is not None

    @property
    def parent(self) -> Optional[StoragePartition]:
        return self._parent

    @property
    def level(self) -> int:
        return self._level

    @property
    def color(self) -> Optional[Shape]:
        return self._color

    def get_root(self) -> Storage:
        if self._parent is None:
            return self
        else:
            return self._parent.get_root()

    def volume(self) -> int:
        return self.extents.volume()

    def overlaps(self, other: Storage) -> bool:
        if self is other:
            return True

        lhs = self
        rhs = other

        lhs_root = lhs.get_root()
        rhs_root = rhs.get_root()

        if lhs_root is not rhs_root:
            return False

        lhs_lvl = lhs.level
        rhs_lvl = rhs.level

        if lhs_lvl > rhs_lvl:
            lhs, rhs = rhs, lhs
            lhs_lvl, rhs_lvl = rhs_lvl, lhs_lvl

        while lhs_lvl < rhs_lvl:
            rhs_parent = rhs.parent
            assert rhs_parent is not None
            rhs = rhs_parent.parent
            rhs_lvl -= 2

        if lhs is rhs:
            return True
        else:
            assert lhs.has_parent and rhs.has_parent
            assert self.parent is not None
            # Legion doesn't allow passing aliased partitions to a task
            if lhs.parent is not rhs.parent:
                return True
            else:
                # TODO: This check is incorrect if the partition is aliased.
                #       Since we only have a tiling, which is a disjoint
                #       partition, we put this assertion here to remember
                #       that we need to exdtend this logic if we have other
                #       partitions. (We need to carry around the disjointness
                #       of each partition.)
                assert isinstance(self.parent._partition, Tiling)
                return lhs.color == rhs.color

    def attach_external_allocation(
        self, context: Context, alloc: Attachable, share: bool
    ) -> None:
        # If the storage has not been set, and this is a non-temporary
        # singleton attachment, we can reuse an existing RegionField that was
        # previously attached to this buffer.
        # This is the only situation where we can attach the same buffer to
        # two Stores, since they are both backed by the same RegionField.
        if self._data is None and share and isinstance(alloc, memoryview):
            self._data = attachment_manager.reuse_existing_attachment(alloc)
            if self._data is not None:
                return
        # Force the RegionField to be instantiated, do the attachment normally
        assert isinstance(self.data, RegionField)
        self.data.attach_external_allocation(context, alloc, share)

    def slice(self, tile_shape: Shape, offsets: Shape) -> Storage:
        if self.kind is Future:
            return self

        # As an interesting optimization, if we can evenly tile the
        # region in all dimensions of the parent region, then we'll make
        # a disjoint tiled partition with as many children as possible
        shape = self.get_root().extents
        can_tile_completely = (offsets % tile_shape).sum() == 0 and (
            shape % tile_shape
        ).sum() == 0

        if can_tile_completely and partition_manager.use_complete_tiling(
            shape, tile_shape
        ):
            color_shape = shape // tile_shape
            color = offsets // tile_shape
            offsets = Shape((0,) * shape.ndim)
            complete = True
        else:
            color_shape = Shape((1,) * shape.ndim)
            color = Shape((0,) * shape.ndim)
            complete = False

        tiling = Tiling(tile_shape, color_shape, offsets)
        # We create a slice partition directly off of the root
        partition = StoragePartition(
            1,
            self.get_root(),
            tiling,
            complete=complete,
        )
        return partition.get_child(color)

    def partition(self, partition: PartitionBase) -> StoragePartition:
        complete = partition.is_complete_for(self.extents, self.offsets)
        return StoragePartition(
            self._level + 1, self, partition, complete=complete
        )

    def get_inline_allocation(
        self,
        shape: Shape,
        context: Optional[Context] = None,
        transform: Optional[AffineTransform] = None,
    ) -> InlineMappedAllocation:
        assert isinstance(self.data, RegionField)
        return self.data.get_inline_allocation(
            shape, context=context, transform=transform
        )

    def find_key_partition(
        self, restrictions: tuple[Restriction, ...]
    ) -> Optional[PartitionBase]:
        if (
            self._key_partition is not None
            and self._key_partition.satisfies_restriction(restrictions)
        ):
            return self._key_partition
        elif self._parent is not None:
            return self._parent.find_key_partition(restrictions)
        else:
            return None

    def set_key_partition(self, partition: PartitionBase) -> None:
        self._key_partition = partition

    def reset_key_partition(self) -> None:
        self._key_partition = None

    def find_or_create_legion_partition(
        self,
        functor: PartitionBase,
        complete: bool,
        color_shape: Optional[Shape] = None,
        color_transform: Optional[Transform] = None,
    ) -> Optional[LegionPartition]:
        if self.kind is not RegionField:
            return None

        assert isinstance(self.data, RegionField)
        assert color_shape is None or color_transform is not None

        if functor in self._partitions and color_shape is None:
            return self._partitions[functor]

        part = functor.construct(
            self.data.region,
            complete=complete,
            color_shape=color_shape,
            color_transform=color_transform,  # type: ignore
        )
        if color_shape is None:
            self._partitions[functor] = part

        return part


class StorePartition:
    def __init__(
        self,
        store: Store,
        partition: PartitionBase,
        storage_partition: StoragePartition,
    ) -> None:
        self._store = store
        self._partition = partition
        self._storage_partition = storage_partition

    @property
    def store(self) -> Store:
        return self._store

    @property
    def partition(self) -> PartitionBase:
        return self._partition

    @property
    def transform(self) -> TransformStackBase:
        return self._store.transform

    def get_child_store(self, *indices: int) -> Store:
        color = self.transform.invert_color(Shape(indices))
        child_storage = self._storage_partition.get_child(color)
        child_transform = self.transform
        for dim, offset in enumerate(child_storage.offsets):
            child_transform = TransformStack(
                Shift(dim, -offset), child_transform
            )
        return Store(
            self._store.type,
            child_storage,
            child_transform,
            shape=child_storage.extents,
        )

    def get_requirement(
        self,
        launch_ndim: int,
        proj_fn: Optional[Union[ProjFn, int]] = None,
    ) -> Proj:
        part = self._storage_partition.find_or_create_legion_partition()
        if part is not None:
            if isinstance(proj_fn, int):
                proj_id = proj_fn
            else:
                proj_id = self._store.compute_projection(proj_fn, launch_ndim)
                if self._partition.needs_delinearization(launch_ndim):
                    assert proj_id == 0
                    proj_id = runtime.get_delinearize_functor()
        else:
            proj_id = 0
        return self._partition.requirement(part, proj_id)

    def is_disjoint_for(self, launch_domain: Optional[Rect]) -> bool:
        return self._storage_partition.is_disjoint_for(launch_domain)


class Store:
    def __init__(
        self,
        dtype: _Dtype,
        storage: Storage,
        transform: Optional[TransformStackBase] = None,
        shape: Optional[Shape] = None,
        ndim: Optional[int] = None,
    ) -> None:
        """
        Unlike in Arrow where all data is backed by objects that
        implement the Python Buffer protocol, in Legate data is backed
        by Legate Store objects. A Store is a logical unit of storage
        backed by one of three containers: RegionField, Future, or FutureMap.
        Backing storages are materialized lazily and may not be created, if
        the lazy evaluation logic optimized them away.

        Parameters
        ----------
        runtime : legate.core.Runtime
            The Legate runtime
        shape : legate.core.Shape
            A Shape object representing the shape of this store
        dtype : legate.core.types._DType
            Data type of this store
        storage : Storage
            A backing storage of this store
        trasnform : TransformStack
            A stack of transforms that describe a view to the storage

        """
        if transform is not None:
            sanitized_transform = transform
        else:
            sanitized_transform = identity
        assert isinstance(shape, Shape) or shape is None
        self._shape = shape
        self._ndim = ndim
        self._dtype = dtype
        self._storage = storage
        self._transform: TransformStackBase = sanitized_transform
        self._key_partition: Union[None, PartitionBase] = None
        # This is a cache for the projection functor id
        # when no custom functor is given
        self._projection: Union[None, int] = None
        self._restrictions: Union[None, tuple[Restriction, ...]] = None
        # We maintain a version on store objects to cache dependent
        # partitions created from the store. Operations that write
        # to stores will bump their version and invalidate dependent
        # partitions that were created with this store as the source.
        self._version = 0

        if self._shape is not None:
            if any(extent < 0 for extent in self._shape.extents):
                raise ValueError(f"Invalid shape: {self._shape}")
        if dtype.variable_size:
            raise NotImplementedError(
                f"Store does not yet support variable size type {dtype}"
            )

    @property
    def linear(self) -> bool:
        return self._storage.linear

    def set_linear(self) -> None:
        return self._storage.set_linear()

    def move_data(self, other: Store) -> None:
        self._storage.move_data(other._storage)

    @property
    def shape(self) -> Shape:
        if self._shape is None:
            # If someone wants to access the shape of an unbound
            # store before it is set, that means the producer task is
            # sitting in the queue, so we should flush the queue.
            runtime.flush_scheduling_window()

            # If the shape is still None, this store has not been passed to any
            # task yet. Accessing the shape of an uninitialized store is
            # illegal.
            if self._shape is None:
                raise ValueError(
                    "Accessing the shape of an uninitialized unbound store "
                    "is illegal. This error usually happens when you try to "
                    "transform the uninitialized unbound store before you "
                    "pass it to any task."
                )
        return self._shape

    @property
    def ndim(self) -> int:
        if self._shape is None:
            assert self._ndim is not None
            return self._ndim
        else:
            return self._shape.ndim

    @property
    def size(self) -> int:
        return prod(self.shape) if self.ndim > 0 else 1

    @property
    def type(self) -> _Dtype:
        """
        Return the type of the data in this storage primitive
        """
        return self._dtype

    def get_dtype(self) -> _Dtype:
        return self._dtype

    @property
    def kind(self) -> Union[Type[RegionField], Type[Future]]:
        """
        Return the type of the Legion storage object backing the data in this
        storage object: either Future, or RegionField.
        """
        return self._storage.kind

    @property
    def unbound(self) -> bool:
        return self._shape is None

    @property
    def scalar(self) -> bool:
        return self.kind is Future and self.shape.volume() == 1

    @property
    def storage(self) -> Union[RegionField, Future]:
        """
        Return the Legion container backing this Store.
        """
        if self.unbound:
            raise RuntimeError(
                "Storage of a variable size store cannot be retrieved "
                "until it is passed to an operation"
            )
        return self._storage.data

    @property
    def has_storage(self) -> bool:
        return self._storage.has_data

    def same_root(self, rhs: Store) -> bool:
        return self._storage.get_root() is rhs._storage.get_root()

    @property
    def extents(self) -> Shape:
        return self._storage.extents

    @property
    def transform(self) -> TransformStackBase:
        return self._transform

    @property
    def transformed(self) -> bool:
        return not self._transform.bottom

    @property
    def version(self) -> int:
        return self._version

    def bump_version(self) -> None:
        self._version += 1

    def attach_external_allocation(
        self, context: Context, alloc: Attachable, share: bool
    ) -> None:
        if not isinstance(alloc, (memoryview, DistributedAllocation)):
            raise ValueError(
                f"Only a memoryview or DistributedAllocation object can be "
                f"attached, but got {alloc}"
            )
        elif self._storage.has_parent:
            raise ValueError("Can only attach buffers to top-level Stores")
        elif self.kind is not RegionField:
            raise ValueError(
                "Can only attach buffers to RegionField-backed Stores"
            )
        elif self.unbound:
            raise ValueError("Cannot attach buffers to variable-size stores")

        self._storage.attach_external_allocation(context, alloc, share)

    def has_fake_dims(self) -> bool:
        return self._transform.adds_fake_dims()

    def comm_volume(self) -> int:
        return self._storage.volume()

    def set_storage(self, data: Union[RegionField, Future]) -> None:
        assert not self.linear
        self._storage.set_data(data)
        if self._shape is None:
            assert isinstance(data, RegionField)
            self._shape = data.shape
            self._storage.set_extents(self._shape)
            self._ndim = None
        else:
            assert isinstance(data, Future)

    def invert_partition(self, partition: PartitionBase) -> PartitionBase:
        return self._transform.invert_partition(partition)

    def __str__(self) -> str:
        return (
            f"Store("
            f"shape: {self._shape}, "
            f"ndim: {self._ndim}, "
            f"type: {self._dtype}, "
            f"storage: {self._storage}, "
            f"transform: {self._transform})"
        )

    def __repr__(self) -> str:
        return str(self)

    # Convert a store in N-D space to that in (N+1)-D space.
    # The extra_dim specifies the added dimension
    def promote(self, extra_dim: int, dim_size: int = 1) -> Store:
        extra_dim = extra_dim + self.ndim if extra_dim < 0 else extra_dim
        if extra_dim < 0 or extra_dim > self.ndim:
            raise ValueError(
                f"Invalid promotion on dimension {extra_dim} for a "
                f"{self.ndim}-D store"
            )

        transform = Promote(extra_dim, dim_size)
        old_shape = self.shape
        shape = transform.compute_shape(old_shape)
        if old_shape == shape:
            return self
        return Store(
            self._dtype,
            self._storage,
            TransformStack(transform, self._transform),
            shape=shape,
        )

    # Take a hyperplane of an N-D store for a given index
    # to create an (N-1)-D store
    def project(self, dim: int, index: int) -> Store:
        dim = dim + self.ndim if dim < 0 else dim
        if dim < 0 or dim >= self.ndim:
            raise ValueError(
                f"Invalid projection on dimension {dim} for a {self.ndim}-D "
                "store"
            )

        old_shape = self.shape
        if index < 0 or index >= old_shape[dim]:
            raise ValueError(
                f"Out-of-bounds projection on dimension {dim} with index "
                f"{index} for a store of shape {old_shape}"
            )

        transform = Project(dim, index)
        shape = transform.compute_shape(old_shape)
        if old_shape == shape:
            return self

        tile_shape = old_shape.update(dim, 1)
        offsets = Shape((0,) * self.ndim).update(dim, index)

        if self.size == 0:
            storage = self._storage
        else:
            storage = self._storage.slice(
                self._transform.invert_extent(tile_shape),
                self._transform.invert_point(offsets),
            )
        return Store(
            self._dtype,
            storage,
            TransformStack(transform, self._transform),
            shape=shape,
        )

    def slice(self, dim: int, sl: slice) -> Store:
        dim = dim + self.ndim if dim < 0 else dim
        if dim < 0 or dim >= self.ndim:
            raise ValueError(
                f"Invalid slicing of dimension {dim} for a {self.ndim}-D store"
            )

        size = self.shape[dim]

        if size == 0 or sl == slice(None):
            return self

        start = 0 if sl.start is None else sl.start
        stop = size if sl.stop is None else sl.stop
        start = start + size if start < 0 else start
        stop = stop + size if stop < 0 else stop
        step = 1 if sl.step is None else sl.step

        if step != 1:
            raise NotImplementedError(f"Unsupported slicing: {sl}")
        elif start >= size or stop > size:
            raise ValueError(
                f"Out-of-bounds slicing {sl} on dimension {dim} for a store "
                f"of shape {self.shape}"
            )

        shape = self.shape
        tile_shape = shape.update(dim, stop - start)
        if shape == tile_shape:
            return self

        offsets = Shape((0,) * self.ndim).update(dim, start)

        storage = self._storage.slice(
            self._transform.invert_extent(tile_shape),
            self._transform.invert_point(offsets),
        )
        transform = (
            self._transform
            if start == 0
            else TransformStack(Shift(dim, -start), self._transform)
        )
        return Store(
            self._dtype,
            storage,
            transform,
            shape=tile_shape,
        )

    def transpose(self, axes: tuple[int, ...]) -> Store:
        if len(axes) != self.ndim:
            raise ValueError(
                f"dimension mismatch: expected {self.ndim} axes, "
                f"but got {len(axes)}"
            )
        elif len(axes) != len(set(axes)):
            raise ValueError(f"duplicate axes found: {axes}")

        for val in axes:
            if val < 0 or val >= self.ndim:
                raise ValueError(
                    f"invalid axis {val} for a {self.ndim}-D store"
                )

        if all(idx == val for idx, val in enumerate(axes)):
            return self

        transform = Transpose(axes)
        shape = transform.compute_shape(self.shape)
        return Store(
            self._dtype,
            self._storage,
            TransformStack(transform, self._transform),
            shape=shape,
        )

    def delinearize(self, dim: int, shape: tuple[int, ...]) -> Store:
        dim = dim + self.ndim if dim < 0 else dim
        if dim < 0 or dim >= self.ndim:
            raise ValueError(
                f"Invalid delinearization of dimension {dim} "
                f"for a {self.ndim}-D store"
            )

        if len(shape) == 1:
            return self
        s = Shape(shape)
        transform = Delinearize(dim, s)
        old_shape = self.shape
        if old_shape[dim] != s.volume():
            raise ValueError(
                f"Dimension of size {old_shape[dim]} "
                f"cannot be delinearized into {old_shape}"
            )
        new_shape = transform.compute_shape(old_shape)
        return Store(
            self._dtype,
            self._storage,
            TransformStack(transform, self._transform),
            shape=new_shape,
        )

    def get_inline_allocation(
        self, context: Optional[Context] = None
    ) -> InlineMappedAllocation:
        assert self.kind is RegionField
        return self._storage.get_inline_allocation(
            self.shape,
            context=context,
            transform=self._transform.get_inverse_transform(self.shape.ndim),
        )

    def overlaps(self, other: Store) -> bool:
        return self._storage.overlaps(other._storage)

    def serialize(self, buf: BufferBuilder) -> None:
        buf.pack_bool(self.kind is Future)
        buf.pack_bool(self.unbound)
        buf.pack_32bit_int(self.ndim)
        buf.pack_32bit_int(self._dtype.code)
        self._transform.serialize(buf)

    def get_key_partition(self) -> Optional[PartitionBase]:
        # Flush outstanding operations to have the key partition of this store
        # registered correctly
        runtime.flush_scheduling_window()

        restrictions = self.find_restrictions()

        if (
            self._key_partition is not None
            and self._key_partition.satisfies_restriction(restrictions)
            # Similiarly to below, if the color shape of the existing key
            # partition no longer matches the number of processors available
            # in the system, then we don't want to return the existing key
            # partition.
            and (
                (self._key_partition.color_shape is None and runtime.num_procs == 1) or
                (self._key_partition.color_shape is not None and self._key_partition.color_shape.volume() == runtime.num_procs)
            )
        ):
            return self._key_partition

        return None

    def has_key_partition(self, restrictions: tuple[Restriction, ...]) -> bool:
        restrictions = self._transform.invert_restrictions(restrictions)
        part = self._storage.find_key_partition(restrictions)
        if part is None:
            return False
        if part.even or self._transform.bottom:
            # If the number of available processors has changed, the existing
            # key partition of this store might be invalid. In that case, we'll
            # return that the key partition is no longer valid, causing it to
            # be recomputed.
            if (part.color_shape is None and runtime.num_procs == 1) or (part.color_shape is not None and part.color_shape.volume() == runtime.num_procs):
                return True
            return False
        return False

        # return (part is not None) and (part.even or self._transform.bottom)

    def set_key_partition(self, partition: PartitionBase) -> None:
        self._key_partition = partition
        # We also update the storage's key partition for other stores
        # sharing the same storage
        self._storage.set_key_partition(
            self._transform.invert_partition(partition)
        )

    def reset_key_partition(self) -> None:
        self._storage.reset_key_partition()

    def compute_key_partition(
        self, restrictions: tuple[Restriction, ...]
    ) -> PartitionBase:
        if (
            self._key_partition is not None
            and self._key_partition.satisfies_restriction(restrictions)
            # Similiarly to above, if the color shape of the existing key
            # partition no longer matches the number of processors available
            # in the system, then we don't want to return the existing key
            # partition.
            and (
                (self._key_partition.color_shape is None and runtime.num_procs == 1) or
                (self._key_partition.color_shape is not None and self._key_partition.color_shape.volume() == runtime.num_procs)
            )
        ):
            return self._key_partition

        # If this is effectively a scalar store, we don't need to partition it
        if self.kind is Future or self.ndim == 0:
            return REPLICATE

        # We need the transformations to be convertible so that we can map
        # the storage partition to this store's coordinate space
        if self._transform.convertible:
            partition = self._storage.find_key_partition(
                self._transform.invert_restrictions(restrictions)
            )
        else:
            partition = None

        # Don't use the key partition of the underlying storage if it doesn't
        # match the number of processors available to the runtime.
        if partition is not None:
            if partition.color_shape is None and runtime.num_procs != 1:
                partition = None
            elif partition.color_shape is not None and runtime.num_procs != partition.color_shape.volume():
                partition = None

        if partition is not None and (
            not (self.transformed and isinstance(partition, Weighted))
        ):
            partition = self._transform.convert_partition(partition)
            return partition
        else:
            launch_shape = partition_manager.compute_launch_shape(
                self,
                restrictions,
            )
            if launch_shape is None:
                partition = REPLICATE
            else:
                tile_shape = partition_manager.compute_tile_shape(
                    self.shape, launch_shape
                )
                partition = Tiling(tile_shape, launch_shape)
            return partition

    def compute_projection(
        self,
        proj_fn: Optional[ProjFn] = None,
        launch_ndim: Optional[int] = None,
    ) -> int:
        assert proj_fn is None or launch_ndim is not None
        # Handle the most common case before we do any analysis
        if self._transform.bottom and proj_fn is None:
            return 0
        # If the store is transformed in some way, we need to compute
        # find the right projection functor that maps points in the color
        # space of the child's partition to subregions of the converted
        # partition

        # For the next common case, we cache the projection functor id
        if proj_fn is None:
            if self._projection is None:
                point = execute_functor_symbolically(self.ndim)
                point = self._transform.invert_symbolic_point(point)
                self._projection = runtime.get_projection(self.ndim, point)
            return self._projection
        # For more general cases, don't bother to cache anything
        else:
            assert launch_ndim is not None
            point = execute_functor_symbolically(launch_ndim, proj_fn)
            point = self._transform.invert_symbolic_point(point)
            return runtime.get_projection(launch_ndim, point)

    def find_restrictions(self) -> tuple[Restriction, ...]:
        if self._restrictions is not None:
            return self._restrictions
        base = (Restriction.UNRESTRICTED,) * self._storage.ndim
        self._restrictions = self._transform.convert_restrictions(base)
        return self._restrictions

    def find_or_create_legion_partition(
        self,
        partition: PartitionBase,
        complete: bool = False,
        preserve_colors: bool = False,
    ) -> Optional[LegionPartition]:
        # Create a Legion partition for a given functor.
        # Before we do that, we need to map the partition back
        # to the original coordinate space.
        if preserve_colors:
            return self._storage.find_or_create_legion_partition(
                self._transform.invert_partition(partition),
                complete=complete,
                color_shape=partition.color_shape,
                color_transform=self._transform.get_inverse_color_transform(  # type: ignore # noqa
                    partition.color_shape.ndim,  # type: ignore
                ),
            )
        else:
            return self._storage.find_or_create_legion_partition(
                self._transform.invert_partition(partition),
                complete=complete,
            )

    def partition(self, partition: PartitionBase) -> StorePartition:
        storage_partition = self._storage.partition(
            self.invert_partition(partition),
        )
        return StorePartition(self, partition, storage_partition)

    def partition_by_tiling(
        self, tile_shape: Union[Shape, Sequence[int]]
    ) -> StorePartition:
        if self.unbound:
            raise TypeError("Unbound store cannot be manually partitioned")
        if not isinstance(tile_shape, Shape):
            tile_shape = Shape(tile_shape)
        launch_shape = (self.shape + tile_shape - 1) // tile_shape
        partition = Tiling(tile_shape, launch_shape)
        return self.partition(partition)
