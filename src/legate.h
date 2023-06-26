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

/**
 * @mainpage Legate C++ API reference
 *
 * This is an API reference for Legate's C++ components.
 */

#include "legion.h"
// legion.h has to go before these
#include "core/data/allocator.h"
#include "core/data/scalar.h"
#include "core/data/store.h"
#include "core/legate_c.h"
#include "core/mapping/mapping.h"
#include "core/runtime/runtime.h"
#include "core/task/registrar.h"
#include "core/task/task.h"
#include "core/type/type_traits.h"
#include "core/utilities/deserializer.h"
#include "core/utilities/dispatch.h"
#include "core/utilities/typedefs.h"
#include "legate_defines.h"
