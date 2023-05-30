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

from legion_cffi import is_legion_python, ffi, lib as legion

if is_legion_python == False:
    from legion_top import (
        legion_canonical_python_main,
        legion_canonical_python_cleanup,
    )
    from ..driver.main import prepare_driver, CanonicalDriver
    import atexit, os, shlex, sys

    argv = ["legate"] + shlex.split(os.environ.get("LEGATE_CONFIG", ""))

    driver = prepare_driver(argv, CanonicalDriver)

    if driver.dry_run:
        sys.exit(0)

    os.environ.update(driver.env)

    legion_canonical_python_main(driver.cmd)
    atexit.register(legion_canonical_python_cleanup)

from ._legion import (
    LEGATE_MAX_DIM,
    LEGATE_MAX_FIELDS,
    Point,
    Rect,
    Domain,
    Transform,
    AffineTransform,
    IndexAttach,
    IndexDetach,
    IndexSpace,
    PartitionFunctor,
    PartitionByDomain,
    PartitionByRestriction,
    PartitionByImage,
    PartitionByImageRange,
    EqualPartition,
    PartitionByWeights,
    IndexPartition,
    FieldSpace,
    FieldID,
    Region,
    Partition,
    Fill,
    IndexFill,
    Copy,
    IndexCopy,
    Attach,
    Detach,
    Acquire,
    Release,
    Future,
    OutputRegion,
    PhysicalRegion,
    InlineMapping,
    Task,
    FutureMap,
    IndexTask,
    Fence,
    ArgumentMap,
    BufferBuilder,
    legate_task_preamble,
    legate_task_progress,
    legate_task_postamble,
)

# Import select types for Legate library construction
from .allocation import DistributedAllocation
from .legate import (
    Array,
    Field,
    Library,
    Table,
)
from .machine import EmptyMachineError, Machine, ProcessorKind, ProcessorSlice
from .runtime import (
    Annotation,
    get_legate_runtime,
    get_legion_context,
    get_legion_runtime,
    get_machine,
    legate_add_library,
    track_provenance,
)
# TODO (rohany): This might be too bold...
from .store import ExternalStoreReference as Store

from .types import (
    array_type,
    bool_,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float16,
    float32,
    float64,
    complex64,
    complex128,
    struct_type,
    Dtype,
    ReductionOp,
)
from .io import CustomSplit, TiledSplit, ingest
