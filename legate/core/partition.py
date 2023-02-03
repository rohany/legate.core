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

from abc import ABC, abstractmethod, abstractproperty
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Optional, Sequence, Type, Union
import weakref

from . import (
    BufferBuilder,
    FutureMap,
    IndexPartition,
    PartitionByDomain,
    PartitionByImage,
    PartitionByImageRange,
    PartitionByPreimage,
    PartitionByPreimageRange,
    PartitionByRestriction,
    PartitionByWeights,
    Point,
    Rect,
    Transform,
    ffi,
    legion,
)
from .launcher import Broadcast, Partition
from .restriction import Restriction
from .runtime import runtime
from .shape import Shape

if TYPE_CHECKING:
    from . import Partition as LegionPartition, Region


RequirementType = Union[Type[Broadcast], Type[Partition]]


def _mapper_argument() -> bytes:
    argbuf = BufferBuilder()
    runtime.machine.pack(argbuf)
    argbuf.pack_32bit_uint(runtime.get_sharding(0))
    return argbuf.get_string()


class PartitionBase(ABC):
    @abstractproperty
    def color_shape(self) -> Optional[Shape]:
        ...

    @abstractproperty
    def even(self) -> bool:
        ...

    @abstractmethod
    def construct(
        self,
        region: Region,
        complete: bool = False,
        color_shape: Optional[Shape] = None,
        color_transform: Optional[Transform] = None,
    ) -> Optional[LegionPartition]:
        ...

    @abstractmethod
    def is_complete_for(self, extents: Shape, offsets: Shape) -> bool:
        ...

    @abstractmethod
    def is_disjoint_for(self, launch_domain: Optional[Rect]) -> bool:
        ...

    @abstractmethod
    def satisfies_restriction(
        self, restrictions: Sequence[Restriction]
    ) -> bool:
        ...

    @abstractmethod
    def needs_delinearization(self, launch_ndim: int) -> bool:
        ...

    @abstractproperty
    def requirement(self) -> RequirementType:
        ...

    def get_stores(self) -> list[Any]:
        return []

    def valid(self) -> bool:
        return True


class Replicate(PartitionBase):
    @property
    def color_shape(self) -> Optional[Shape]:
        return None

    @property
    def even(self) -> bool:
        return True

    @property
    def requirement(self) -> RequirementType:
        return Broadcast

    def is_complete_for(self, extents: Shape, offsets: Shape) -> bool:
        return True

    def is_disjoint_for(self, launch_domain: Optional[Rect]) -> bool:
        return launch_domain is None

    def __hash__(self) -> int:
        return hash(self.__class__)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Replicate)

    def __str__(self) -> str:
        return "Replicate"

    def __repr__(self) -> str:
        return str(self)

    def needs_delinearization(self, launch_ndim: int) -> bool:
        return False

    def satisfies_restriction(
        self, restrictions: Sequence[Restriction]
    ) -> bool:
        return True

    def translate(self, offset: float) -> Replicate:
        return self

    def translate_range(self, offset: float) -> Replicate:
        return self

    def scale(self, scale: tuple[int]) -> Replicate:
        return self

    def construct(
        self,
        region: Region,
        complete: bool = False,
        color_shape: Optional[Shape] = None,
        color_transform: Optional[Transform] = None,
    ) -> Optional[LegionPartition]:
        return None


REPLICATE = Replicate()


class Interval:
    def __init__(self, lo: int, extent: int) -> None:
        self._lo = lo
        self._hi = lo + extent

    def overlaps(self, other: Interval) -> bool:
        return not (other._hi <= self._lo or self._hi <= other._lo)


class Tiling(PartitionBase):
    def __init__(
        self,
        tile_shape: Shape,
        color_shape: Shape,
        offset: Optional[Shape] = None,
    ):
        assert len(tile_shape) == len(color_shape)
        self._tile_shape = tile_shape
        self._color_shape = color_shape
        self._offset = (
            Shape((0,) * len(tile_shape)) if offset is None else offset
        )
        self._hash: Union[int, None] = None

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Tiling)
            and self._tile_shape == other._tile_shape
            and self._color_shape == other._color_shape
            and self._offset == other._offset
        )

    @property
    def tile_shape(self) -> Shape:
        return self._tile_shape

    @property
    def color_shape(self) -> Optional[Shape]:
        return self._color_shape

    @property
    def even(self) -> bool:
        return True

    @property
    def requirement(self) -> RequirementType:
        return Partition

    @property
    def offset(self) -> Shape:
        return self._offset

    def __hash__(self) -> int:
        if self._hash is not None:
            return self._hash

        self._hash = hash(
            (
                self.__class__,
                self._tile_shape,
                self._color_shape,
                self._offset,
            )
        )
        return self._hash

    def __str__(self) -> str:
        return (
            f"Tiling(tile:{self._tile_shape}, "
            f"color:{self._color_shape}, "
            f"offset:{self._offset})"
        )

    def __repr__(self) -> str:
        return str(self)

    def needs_delinearization(self, launch_ndim: int) -> bool:
        return launch_ndim != self._color_shape.ndim

    def satisfies_restriction(
        self, restrictions: Sequence[Restriction]
    ) -> bool:
        for dim, restriction in enumerate(restrictions):
            if (
                restriction == Restriction.RESTRICTED
                and self._color_shape[dim] > 1
            ):
                return False
        return True

    def is_complete_for(self, extents: Shape, offsets: Shape) -> bool:
        my_lo = self._offset
        my_hi = self._offset + self.tile_shape * self._color_shape

        return my_lo <= offsets and offsets + extents <= my_hi

    def is_disjoint_for(self, launch_domain: Optional[Rect]) -> bool:
        return (
            launch_domain is None
            or launch_domain.get_volume() <= self._color_shape.volume()
        )

    def has_color(self, color: Shape) -> bool:
        return color >= 0 and color < self._color_shape

    @lru_cache
    def get_subregion_size(self, extents: Shape, color: Shape) -> Shape:
        lo = self._tile_shape * color + self._offset
        hi = self._tile_shape * (color + 1) + self._offset
        lo = Shape(max(0, coord) for coord in lo)
        hi = Shape(min(max, coord) for (max, coord) in zip(extents, hi))
        return Shape(hi - lo)

    def get_subregion_offsets(self, color: Shape) -> Shape:
        return self._tile_shape * color + self._offset

    def translate(self, offset: Shape) -> Tiling:
        return Tiling(
            self._tile_shape,
            self._color_shape,
            self._offset + offset,
        )

    # This function promotes the translated partition to Replicate if it
    # doesn't overlap with the original partition.
    def translate_range(self, offset: Shape) -> Union[Replicate, Tiling]:
        promote = False
        for ext, off in zip(self._tile_shape, offset):
            mine = Interval(0, ext)
            other = Interval(off, ext)
            if not mine.overlaps(other):
                promote = True
                break

        if promote:
            # TODO: We can actually bloat the tile so that all stencils within
            #       the range are contained, but here we simply replicate
            #       the region, as this usually happens for small inputs.
            return REPLICATE
        else:
            return Tiling(
                self._tile_shape,
                self._color_shape,
                self._offset + offset,
            )

    def scale(self, scale: tuple[int]) -> Tiling:
        if self._offset.volume() > 0:
            raise ValueError(
                "scaling of a tile partition with non-zero offsets is "
                "not well-defined"
            )
        return Tiling(
            self._tile_shape * scale,
            self._color_shape,
            self._offset * scale,
        )

    def construct(
        self,
        region: Region,
        complete: bool = False,
        color_shape: Optional[Shape] = None,
        color_transform: Optional[Transform] = None,
    ) -> Optional[LegionPartition]:
        assert color_shape is None or color_transform is not None
        index_space = region.index_space
        index_partition = runtime.partition_manager.find_index_partition(
            index_space, self, color_shape=color_shape
        )
        if index_partition is None:
            tile_shape = self._tile_shape
            transform = Transform(tile_shape.ndim, tile_shape.ndim)
            for idx, size in enumerate(tile_shape):
                transform.trans[idx, idx] = size

            lo = Shape((0,) * tile_shape.ndim) + self._offset
            hi = self._tile_shape - 1 + self._offset

            extent = Rect(hi, lo, exclusive=False)

            color_space = runtime.find_or_create_index_space(
                self._color_shape if color_shape is None else color_shape
            )

            if color_transform is not None:
                transform = color_transform.compose(transform)

            functor = PartitionByRestriction(transform, extent)
            if complete:
                kind = (
                    legion.LEGION_DISJOINT_COMPLETE_KIND
                    if color_shape is None
                    else legion.LEGION_ALIASED_COMPLETE_KIND  # type: ignore
                )
            else:
                kind = (
                    legion.LEGION_DISJOINT_INCOMPLETE_KIND
                    if color_shape is None
                    else legion.LEGION_ALIASED_INCOMPLETE_KIND  # type: ignore
                )
            index_partition = IndexPartition(
                runtime.legion_context,
                runtime.legion_runtime,
                index_space,
                color_space,
                functor,
                kind=kind,
                keep=True,  # export this partition functor to other libraries
            )
            if self._offset == Shape((0,) * len(self._tile_shape)):
                runtime.partition_manager.record_index_partition(
                    index_space, self, index_partition, color_shape=color_shape
                )
        return region.get_child(index_partition)


class Weighted(PartitionBase):
    def __init__(self, color_shape: Shape, weights: FutureMap) -> None:
        self._color_shape = color_shape
        self._weights = weights
        self._hash: Union[int, None] = None

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Weighted)
            and self._color_shape == other._color_shape
            and self._weights == other._weights
        )

    @property
    def color_shape(self) -> Optional[Shape]:
        return self._color_shape

    @property
    def even(self) -> bool:
        return False

    @property
    def requirement(self) -> RequirementType:
        return Partition

    def __hash__(self) -> int:
        if self._hash is not None:
            return self._hash

        self._hash = hash(
            (
                self.__class__,
                self._color_shape,
                self._weights,
            )
        )
        return self._hash

    def __str__(self) -> str:
        return (
            f"Weighted(color:{self._color_shape}, " f"weights:{self._weights})"
        )

    def __repr__(self) -> str:
        return str(self)

    def needs_delinearization(self, launch_ndim: int) -> bool:
        return launch_ndim != self._color_shape.ndim

    def satisfies_restriction(
        self, restrictions: Sequence[Restriction]
    ) -> bool:
        return all(
            restriction != Restriction.RESTRICTED
            for restriction in restrictions
        )

    def is_complete_for(self, extents: Shape, offsets: Shape) -> bool:
        # Weighted partition is complete by definition
        return True

    def is_disjoint_for(self, launch_domain: Optional[Rect]) -> bool:
        # Weighted partition is disjoint by definition
        return True

    def has_color(self, color: Shape) -> bool:
        return color >= 0 and color < self._color_shape

    def translate(self, offset: Shape) -> None:
        raise NotImplementedError("This method shouldn't be invoked")

    def translate_range(self, offset: Shape) -> None:
        raise NotImplementedError("This method shouldn't be invoked")

    def construct(
        self,
        region: Region,
        complete: bool = False,
        color_shape: Optional[Shape] = None,
        color_transform: Optional[Transform] = None,
    ) -> Optional[LegionPartition]:
        assert complete

        index_space = region.index_space
        index_partition = runtime.partition_manager.find_index_partition(
            index_space, self
        )
        if index_partition is None:
            color_space = runtime.find_or_create_index_space(self._color_shape)
            functor = PartitionByWeights(self._weights)
            kind = legion.LEGION_DISJOINT_COMPLETE_KIND
            index_partition = IndexPartition(
                runtime.legion_context,
                runtime.legion_runtime,
                index_space,
                color_space,
                functor,
                kind=kind,
                keep=True,  # export this partition functor to other libraries
            )
            runtime.partition_manager.record_index_partition(
                index_space, self, index_partition
            )
        return region.get_child(index_partition)


class ImagePartition(PartitionBase):
    def __init__(
        self,
        store: Any,
        part: PartitionBase,
        mapper: int,
        range: bool = False,
        disjoint: bool = True,
        complete: bool = True,
    ) -> None:
        self._mapper = mapper
        self._store_ref = weakref.ref(store)
        self._storage_id = self._store._storage._unique_id
        self._store_version = self._store._version
        self._part = part
        # Whether this is an image or image_range operation.
        self._range = range
        self._disjoint = disjoint
        self._complete = complete

    @property
    def _store(self):
        return self._store_ref()

    def get_stores(self) -> list[Any]:
        return [self._store]

    def valid(self) -> bool:
        return self._store is not None

    # def make_weakref_gc_callback(self):
    #     def callback(store):
    #         part = self
    #         # Remove the dependent partitions from all referencing structures.
    #         if part in runtime.partition_manager._index_partitions_by_partition_base:
    #             index_part_keys = runtime.partition_manager._index_partitions_by_partition_base[part]
    #             for key in index_part_keys:
    #                 if key in runtime.partition_manager._index_partitions:
    #                     del runtime.partition_manager._index_partitions[key]
    #             del runtime.partition_manager._index_partitions_by_partition_base[part]
    #         if part in runtime.partition_manager._legion_partitions_by_partition_base:
    #             legion_part_keys = runtime.partition_manager._legion_partitions_by_partition_base[part]
    #             for key in legion_part_keys:
    #                 if key in runtime.partition_manager._legion_partitions:
    #                     del runtime.partition_manager._legion_partitions[key]
    #             del runtime.partition_manager._legion_partitions_by_partition_base[part]
    #     return callback

    @property
    def color_shape(self) -> Optional[Shape]:
        return self._part.color_shape

    @property
    def even(self) -> bool:
        return False

    def construct(
        self,
        region: Region,
        complete: bool = False,
        color_shape: Optional[Shape] = None,
        color_transform: Optional[Transform] = None,
    ) -> Optional[LegionPartition]:
        # TODO (rohany): We can't import RegionField due to an import cycle.
        # assert(isinstance(self._store.storage, RegionField))
        source_region = self._store.storage.region
        source_field = self._store.storage.field

        # TODO (rohany): What should the value of complete be?
        source_part = self._store.find_or_create_legion_partition(
            self._part,
            preserve_colors=True,
        )
        if self._range:
            functor = PartitionByImageRange(
                source_region,
                source_part,
                source_field.field_id,
                mapper=self._mapper,
                mapper_arg=_mapper_argument(),
            )
        else:
            functor = PartitionByImage(  # type: ignore
                source_region,
                source_part,
                source_field.field_id,
                mapper=self._mapper,
                mapper_arg=_mapper_argument(),
            )
        index_partition = runtime.partition_manager.find_index_partition(
            region.index_space, self
        )
        if index_partition is None:
            if self._disjoint and self._complete:
                kind = legion.LEGION_DISJOINT_COMPLETE_KIND
            elif self._disjoint and not self._complete:
                kind = legion.LEGION_DISJOINT_INCOMPLETE_KIND
            elif not self._disjoint and self._complete:
                kind = legion.LEGION_ALIASED_COMPLETE_KIND  # type: ignore
            else:
                kind = legion.LEGION_ALIASED_INCOMPLETE_KIND  # type: ignore
            index_partition = IndexPartition(
                runtime.legion_context,
                runtime.legion_runtime,
                region.index_space,
                source_part.color_space,
                functor=functor,
                kind=kind,
                keep=True,
            )
            runtime.partition_manager.record_index_partition(
                region.index_space, self, index_partition
            )
        return region.get_child(index_partition)

    def is_complete_for(self, extents: Shape, offsets: Shape) -> bool:
        return self._complete

    def is_disjoint_for(self, launch_domain: Optional[Rect]) -> bool:
        return self._disjoint

    def satisfies_restriction(
        self, restrictions: Sequence[Restriction]
    ) -> bool:
        for restriction in restrictions:
            # If there are some restricted dimensions to this store,
            # then this key partition is likely not a good choice.
            if restriction == Restriction.RESTRICTED:
                return False
        return True

    def needs_delinearization(self, launch_ndim: int) -> bool:
        assert self.color_shape is not None
        return launch_ndim != self.color_shape.ndim

    @property
    def requirement(self) -> RequirementType:
        return Partition

    def __hash__(self) -> int:
        return hash(
            (
                self.__class__,
                self._storage_id,
                # Importantly, we _cannot_ store the version of the store
                # in the hash value. This is because the store's version may
                # change after we've already put this functor into a table.
                # That would result in the hash value changing without moving
                # the position in the table, breaking invariants of the table.
                # However, we must still check for version in equality to avoid
                # using old values.
                # self._store._version,
                self._part,
                self._range,
                self._mapper,
            )
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ImagePartition)
            # Importantly, we check equality of Storage objects rather than
            # Stores. This is because Stores can have equivalent storages but
            # not be equal due to transformations on the store. By checking
            # that the Storages are equal, we are basically checking whether
            # we have the same RegionField object.
            and self._storage_id == other._storage_id
            and self._store_version == other._store_version
            and self._part == other._part
            and self._range == other._range
            and self._mapper == other._mapper
        )

    def __str__(self) -> str:
        return f"image({self._store}, {self._part}, range={self._range})"

    def __repr__(self) -> str:
        return str(self)


class PreimagePartition(PartitionBase):
    # TODO (rohany): I don't even know if I need a store here. I really just
    #  need the index space that is being partitioned (or the IndexPartition).
    #  For simplicities sake it seems like taking the store is fine.
    def __init__(
        self,
        source: Any,
        dest: Any,
        part: PartitionBase,
        mapper: int,
        range: bool = False,
        disjoint: bool = False,
        complete: bool = True,
    ) -> None:
        self._mapper = mapper
        self._source_ref = weakref.ref(source)
        self._source_storage_id = self._source._storage._unique_id
        # Importantly, we don't store a reference to `dest` and instead
        # hold onto a handle of the underlying region. This is important
        # because if we store dest itself on the partition then legate
        # can't collect and reuse the storage under dest. Since all we
        # actually need from dest is the underlying index space, storing
        # the region sidesteps this limitation.
        self._dest_region = dest.storage.region
        self._part = part
        # Whether this is an image or image_range operation.
        self._range = range
        self._disjoint = disjoint
        self._complete = complete

    @property
    def _source(self):
        return self._source_ref()

    def get_stores(self) -> list[Any]:
        return [self._source]

    @property
    def color_shape(self) -> Optional[Shape]:
        return self._part.color_shape

    @property
    def even(self) -> bool:
        return False

    def construct(
        self,
        region: Region,
        complete: bool = False,
        color_shape: Optional[Shape] = None,
        color_transform: Optional[Transform] = None,
    ) -> Optional[LegionPartition]:
        dest_part = self._part.construct(self._dest_region)
        source_region = self._source.storage.region
        source_field = self._source.storage.field.field_id
        functorFn = (
            PartitionByPreimageRange if self._range else PartitionByPreimage
        )
        functor = functorFn(
            dest_part.index_partition,  # type: ignore
            source_region,
            source_region,
            source_field,
            mapper=self._mapper,
            mapper_arg=_mapper_argument(),
        )
        index_partition = runtime.partition_manager.find_index_partition(
            region.index_space, self
        )
        if index_partition is None:
            if self._disjoint and self._complete:
                kind = legion.LEGION_DISJOINT_COMPLETE_KIND
            elif self._disjoint and not self._complete:
                kind = legion.LEGION_DISJOINT_INCOMPLETE_KIND
            elif not self._disjoint and self._complete:
                kind = legion.LEGION_ALIASED_COMPLETE_KIND  # type: ignore
            else:
                kind = legion.LEGION_ALIASED_INCOMPLETE_KIND  # type: ignore
            # Discharge some typing errors.
            assert dest_part is not None
            index_partition = IndexPartition(
                runtime.legion_context,
                runtime.legion_runtime,
                region.index_space,
                dest_part.color_space,
                functor=functor,
                kind=kind,
                keep=True,
            )
            runtime.partition_manager.record_index_partition(
                region.index_space, self, index_partition
            )
        return region.get_child(index_partition)

    def is_complete_for(self, extents: Shape, offsets: Shape) -> bool:
        return self._complete

    def is_disjoint_for(self, launch_domain: Optional[Rect]) -> bool:
        return self._disjoint

    def satisfies_restriction(
        self, restrictions: Sequence[Restriction]
    ) -> bool:
        for restriction in restrictions:
            if restriction != Restriction.UNRESTRICTED:
                raise NotImplementedError
        return True

    def needs_delinearization(self, launch_ndim: int) -> bool:
        assert self.color_shape is not None
        return launch_ndim != self.color_shape.ndim

    @property
    def requirement(self) -> RequirementType:
        return Partition

    def __hash__(self) -> int:
        return hash(
            (
                self.__class__,
                self._source_storage_id,
                # Importantly, we _cannot_ store the version of the store
                # in the hash value. This is because the store's version may
                # change after we've already put this functor into a table.
                # That would result in the hash value changing without moving
                # the position in the table, breaking invariants of the table.
                # However, we must still check for version in equality to avoid
                # using old values.
                # self._store._version,
                self._dest_region.index_space,
                self._part,
                self._range,
                self._mapper,
            )
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, PreimagePartition)
            # See the comment on ImagePartition.__eq__ about why we use
            # source._storage for equality.
            and self._source_storage_id == other._source_storage_id
            and self._source._version == other._source._version
            and self._dest_region.index_space == other._dest_region.index_space
            and self._part == other._part
            and self._range == other._range
            and self._mapper == other._mapper
        )

    def __str__(self) -> str:
        return f"preimage({self._source}, {self._part}, range={self._range})"

    def __repr__(self) -> str:
        return str(self)


class DomainPartition(PartitionBase):
    def __init__(
        self,
        shape: Shape,
        color_shape: Shape,
        domains: Union[FutureMap, dict[Point, Rect]],
    ):
        self._color_shape = color_shape
        self._domains = domains
        self._shape = shape
        if len(shape) == 0:
            raise AssertionError

    @property
    def color_shape(self) -> Optional[Shape]:
        return self._color_shape

    @property
    def even(self) -> bool:
        return False

    def construct(
        self,
        region: Region,
        complete: bool = False,
        color_shape: Optional[Shape] = None,
        color_transform: Optional[Transform] = None,
    ) -> Optional[LegionPartition]:
        index_space = region.index_space
        index_partition = runtime.partition_manager.find_index_partition(
            index_space, self
        )
        if index_partition is None:
            functor = PartitionByDomain(self._domains)
            index_partition = IndexPartition(
                runtime.legion_context,
                runtime.legion_runtime,
                index_space,
                runtime.find_or_create_index_space(self._color_shape),
                functor=functor,
                keep=True,
            )
            # runtime.partition_manager.record_index_partition(
            #     index_space, self, index_partition
            # )
        return region.get_child(index_partition)

    # TODO (rohany): We could figure this out by staring at the domain map.
    def is_complete_for(self, extents: Shape, offsets: Shape) -> bool:
        return False

    # TODO (rohany): We could figure this out by staring at the domain map.
    def is_disjoint_for(self, launch_domain: Optional[Rect]) -> bool:
        return False

    def satisfies_restriction(
        self, restrictions: Sequence[Restriction]
    ) -> bool:
        for restriction in restrictions:
            # If there are some restricted dimensions to this store,
            # then this key partition is likely not a good choice.
            if restriction == Restriction.RESTRICTED:
                return False
        return True

    @property
    def requirement(self) -> RequirementType:
        return Partition

    def needs_delinearization(self, launch_ndim: int) -> bool:
        return launch_ndim != self._color_shape.ndim

    def __hash__(self) -> int:
        return hash(
            (
                self.__class__,
                self._shape,
                self._color_shape,
                # TODO (rohany): No better ideas...
                id(self._domains),
            )
        )

    # TODO (rohany): Implement this.
    def __eq__(self, other: object) -> bool:
        return False

    def __str__(self) -> str:
        return f"by_domain({self._color_shape}, {self._domains})"

    def __repr__(self) -> str:
        return str(self)


# AffineProjection is translated from C++ to Python from the DISTAL
# AffineProjection functor. In particular, it encapsulates applying affine
# projections on `DomainPartition` objects.
class AffineProjection:
    # Project each point to the following dimensions of the output point.
    # Passing `None` as an entry in `projs` discards the chosen dimension
    # from the projection.
    def __init__(self, projs: list[Optional[int]]):
        self.projs = projs

    @property
    def dim(self) -> int:
        return len(self.projs)

    def project_point(self, point: Point, output_bound: Point) -> Point:
        output_dim = output_bound.dim
        set_mask = [False] * output_dim
        result = Point(dim=output_dim)
        for i in range(0, self.dim):
            mapTo = self.projs[i]
            if mapTo is None:
                continue
            result[mapTo] = point[i]
            set_mask[mapTo] = True
        # Replace unset indices with their boundaries.
        for i in range(0, output_dim):
            if not set_mask[i]:
                result[i] = output_bound[i]
        return result

    def project_partition(
        self, part: DomainPartition, bounds: Rect, tx_point: Any = None
    ) -> DomainPartition:
        projected = {}
        if isinstance(part._domains, FutureMap):
            for point in Rect(hi=part.color_shape):
                fut = part._domains.get_future(point)
                buf = fut.get_buffer()
                dom = ffi.from_buffer("legion_domain_t*", buf)[0]  # type: ignore # noqa
                lg_rect = getattr(
                    legion, f"legion_domain_get_rect_{dom.dim}d"
                )(dom)
                lo = Point(dim=bounds.dim)
                hi = Point(dim=bounds.dim)
                for i in range(dom.dim):
                    lo[i] = lg_rect.lo.x[i]
                    hi[i] = lg_rect.hi.x[i]
                lo = self.project_point(lo, bounds.lo)
                hi = self.project_point(hi, bounds.hi)
                if tx_point is not None:
                    point = tx_point(point)
                projected[point] = Rect(
                    lo=tuple(lo), hi=tuple(hi), exclusive=False
                )
        else:
            for p, r in part._domains.items():
                lo = self.project_point(r.lo, bounds.lo)
                hi = self.project_point(r.hi, bounds.hi)
                if tx_point is not None:
                    p = tx_point(p)
                projected[p] = Rect(
                    lo=tuple(lo), hi=tuple(hi), exclusive=False
                )
        new_shape = Shape(
            tuple(bounds.hi[idx] + 1 for idx in range(bounds.dim))
        )
        color_shape = part.color_shape
        if tx_point is not None:
            color_shape = Shape(tx_point(color_shape, exclusive=True))
        assert color_shape is not None
        return DomainPartition(new_shape, color_shape, projected)
