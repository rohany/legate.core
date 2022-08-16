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
from enum import IntEnum, unique
from typing import Any, TYPE_CHECKING, Optional, Sequence, Type, Union

from . import (
    IndexPartition,
    PartitionByImage,
    PartitionByImageRange,
    PartitionByRestriction,
    PartitionByWeights,
    Rect,
    Transform,
    legion,
)
from .launcher import Broadcast, Partition
from .shape import Shape

if TYPE_CHECKING:
    from . import FutureMap, Partition as LegionPartition, Region
    from .runtime import Runtime


@unique
class Restriction(IntEnum):
    RESTRICTED = -2
    AVOIDED = -1
    UNRESTRICTED = 1


RequirementType = Union[Type[Broadcast], Type[Partition]]


class PartitionBase(ABC):
    @abstractproperty
    def color_shape(self) -> Optional[Shape]:
        ...

    @abstractproperty
    def even(self) -> bool:
        ...

    @abstractmethod
    def construct(
        self, region: Region, complete: bool = False
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

    @abstractproperty
    def runtime(self) -> Runtime:
        ...


class Replicate(PartitionBase):
    def __init__(self, runtime: Runtime):
        self._runtime = runtime

    @property
    def runtime(self):
        return self._runtime

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
        self, region: Region, complete: bool = False
    ) -> Optional[LegionPartition]:
        return None


class Interval:
    def __init__(self, lo: int, extent: int) -> None:
        self._lo = lo
        self._hi = lo + extent

    def overlaps(self, other: Interval) -> bool:
        return not (other._hi <= self._lo or self._hi <= other._lo)


class Tiling(PartitionBase):
    def __init__(
        self,
        runtime: Runtime,
        tile_shape: Shape,
        color_shape: Shape,
        offset: Optional[Shape] = None,
    ):
        assert len(tile_shape) == len(color_shape)
        self._runtime = runtime
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
    def runtime(self) -> Runtime:
        return self._runtime

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
            self._runtime,
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
            return Replicate(self.runtime)
        else:
            return Tiling(
                self._runtime,
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
            self._runtime,
            self._tile_shape * scale,
            self._color_shape,
            self._offset * scale,
        )

    def construct(
        self, region: Region, complete: bool = False
    ) -> Optional[LegionPartition]:
        index_space = region.index_space
        index_partition = self._runtime.find_partition(index_space, self)
        if index_partition is None:
            tile_shape = self._tile_shape
            transform = Transform(tile_shape.ndim, tile_shape.ndim)
            for idx, size in enumerate(tile_shape):
                transform.trans[idx, idx] = size

            lo = Shape((0,) * tile_shape.ndim) + self._offset
            hi = self._tile_shape - 1 + self._offset

            extent = Rect(hi, lo, exclusive=False)

            color_space = self._runtime.find_or_create_index_space(
                self._color_shape
            )
            functor = PartitionByRestriction(transform, extent)
            if complete:
                kind = legion.LEGION_DISJOINT_COMPLETE_KIND
            else:
                kind = legion.LEGION_DISJOINT_INCOMPLETE_KIND
            index_partition = IndexPartition(
                self._runtime.legion_context,
                self._runtime.legion_runtime,
                index_space,
                color_space,
                functor,
                kind=kind,
                keep=True,  # export this partition functor to other libraries
            )
            self._runtime.record_partition(index_space, self, index_partition)
        return region.get_child(index_partition)


class Weighted(PartitionBase):
    def __init__(
        self, runtime: Runtime, color_shape: Shape, weights: FutureMap
    ) -> None:
        self._runtime = runtime
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
    def runtime(self) -> Runtime:
        return self._runtime

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
        self, region: Region, complete: bool = False
    ) -> Optional[LegionPartition]:
        assert complete

        index_space = region.index_space
        index_partition = self._runtime.find_partition(index_space, self)
        if index_partition is None:
            color_space = self._runtime.find_or_create_index_space(
                self._color_shape
            )
            functor = PartitionByWeights(self._weights)
            kind = legion.LEGION_DISJOINT_COMPLETE_KIND
            index_partition = IndexPartition(
                self._runtime.legion_context,
                self._runtime.legion_runtime,
                index_space,
                color_space,
                functor,
                kind=kind,
                keep=True,  # export this partition functor to other libraries
            )
            self._runtime.record_partition(index_space, self, index_partition)
        return region.get_child(index_partition)


# TODO (rohany): Do we need to have a difference between image and preimage?
class ImagePartition(PartitionBase):
    # TODO (rohany): What's the right type to pass through for the partitions and regions here?
    # store is of type legate.Store. However, we can't import it directly due to an import cycle.
    def __init__(self, runtime: Runtime, store: Any, part: PartitionBase, range : bool = False) -> None:
        self._runtime = runtime
        self._store = store
        self._part = part
        # Whether this is an image or image_range operation.
        self._range = range

    @property
    def color_shape(self) -> Optional[Shape]:
        return self._part.color_shape

    @property
    def even(self) -> bool:
        ...

    def construct(
            self, region: Region, complete: bool = False
    ) -> Optional[LegionPartition]:
        # TODO (rohany): We can't import RegionField due to an import cycle.
        # assert(isinstance(self._store.storage, RegionField))
        source_region = self._store.storage.region
        source_field = self._store.storage.field

        # TODO (rohany): What should the value of complete be?
        source_part = self._part.construct(source_region)
        if self._range:
            functor = PartitionByImageRange(source_region, source_part, source_field.field_id)
        else:
            functor = PartitionByImage(source_region, source_part, source_field.field_id)
        # TODO (rohany): Use some information about the partition to figure out whats going on...
        #  Maybe there should be hints that the user can pass in through the constraints.
        kind = legion.LEGION_DISJOINT_INCOMPLETE_KIND
        # TODO (rohany): Let's just create a new partition each time.
        index_partition = IndexPartition(
            self._runtime.legion_context,
            self._runtime.legion_runtime,
            region.index_space,
            source_part.color_space,
            functor=functor,
            kind=kind,
            keep=True,
        )
        self._runtime.record_partition(region.index_space, self, index_partition)
        return region.get_child(index_partition)

    # TODO (rohany): IDK how we're supposed to know this about an image / image range.
    def is_complete_for(self, extents: Shape, offsets: Shape) -> bool:
        return False

    # TODO (rohany): IDK how we're supposed to know this about an image / image range.
    def is_disjoint_for(self, launch_domain: Optional[Rect]) -> bool:
        return False

    # TODO (rohany): IDK how we're supposed to know this about an image / image range.
    def satisfies_restriction(
            self, restrictions: Sequence[Restriction]
    ) -> bool:
        raise NotImplementedError

    # TODO (rohany): IDK what this means...
    def needs_delinearization(self, launch_ndim: int) -> bool:
        return False

    @property
    def requirement(self) -> RequirementType:
        return Partition

    @property
    def runtime(self) -> Runtime:
        return self._runtime

    # TODO (rohany): Implement...
    def __hash__(self) -> int:
        # TODO (rohany): A problem with this (and then using this as a key in the future) is that
        #  the result of the image partition depends on the values in the region. Once the region
        #  has been updated, the partition is different.
        return hash(self.__class__)

    def __eq__(self, other: object) -> bool:
        raise NotImplementedError

    def __str__(self) -> str:
        return f"image({self._store}, {self._part}, range={self._range})"

    def __repr__(self) -> str:
        return str(self)
