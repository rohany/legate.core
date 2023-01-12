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
#include <mutex>

#include "legion.h"

#include "core/mapping/mapping.h"

namespace legate {
namespace mapping {

// This class represents a set of regions that colocate in an instance
struct RegionGroup {
 public:
  using Region = Legion::LogicalRegion;
  using Domain = Legion::Domain;

 public:
  RegionGroup(const std::set<Region>& regions, const Domain bounding_box);
  RegionGroup(std::set<Region>&& regions, const Domain bounding_box);

 public:
  RegionGroup(const RegionGroup&) = default;
  RegionGroup(RegionGroup&&)      = default;

 public:
  std::vector<Region> get_regions() const;
  bool subsumes(const RegionGroup* other);

 public:
  std::set<Region> regions;
  Domain bounding_box;
  std::map<const RegionGroup*, bool> subsumption_cache;
};

std::ostream& operator<<(std::ostream& os, const RegionGroup& region_group);

struct InstanceSet {
 public:
  using Region       = Legion::LogicalRegion;
  using Instance     = Legion::Mapping::PhysicalInstance;
  using Domain       = Legion::Domain;
  using RegionGroupP = std::shared_ptr<RegionGroup>;

 public:
  struct InstanceSpec {
    InstanceSpec() {}
    InstanceSpec(const Instance& inst, const InstanceMappingPolicy& po) : instance(inst), policy(po)
    {
    }

    Instance instance{};
    InstanceMappingPolicy policy{};
  };

 public:
  bool find_instance(Region region, Instance& result, const InstanceMappingPolicy& policy) const;
  RegionGroupP construct_overlapping_region_group(const Region& region,
                                                  const Domain& domain,
                                                  bool exact) const;

 public:
  std::set<Instance> record_instance(RegionGroupP group,
                                     Instance instance,
                                     const InstanceMappingPolicy& policy);

 public:
  bool erase(Instance inst);

 public:
  size_t get_instance_size() const;

 private:
  void dump_and_sanity_check() const;

 private:
  std::map<RegionGroup*, InstanceSpec> instances_;
  std::map<Legion::LogicalRegion, RegionGroupP> groups_;
};

class ReductionInstanceSet {
 public:
  using Region        = Legion::LogicalRegion;
  using Instance      = Legion::Mapping::PhysicalInstance;
  using Domain        = Legion::Domain;
  using ReductionOpID = Legion::ReductionOpID;

 public:
  struct ReductionInstanceSpec {
    ReductionInstanceSpec() {}
    ReductionInstanceSpec(const ReductionOpID& op,
                          const Instance& inst,
                          const InstanceMappingPolicy& po)
      : redop(op), instance(inst), policy(po)
    {
    }

    ReductionOpID redop{0};
    Instance instance{};
    InstanceMappingPolicy policy{};
  };

 public:
  bool find_instance(ReductionOpID& redop,
                     Region& region,
                     Instance& result,
                     const InstanceMappingPolicy& policy) const;

 public:
  void record_instance(ReductionOpID& redop,
                       Region& region,
                       Instance& instance,
                       const InstanceMappingPolicy& policy);

 private:
  std::map<Region, ReductionInstanceSpec> instances_;
};

class BaseInstanceManager {
 public:
  using Region       = Legion::LogicalRegion;
  using RegionTreeID = Legion::RegionTreeID;
  using Instance     = Legion::Mapping::PhysicalInstance;
  using Domain       = Legion::Domain;
  using FieldID      = Legion::FieldID;
  using Memory       = Legion::Memory;

 public:
  struct FieldMemInfo {
   public:
    FieldMemInfo(RegionTreeID t, FieldID f, Memory m) : tid(t), fid(f), memory(m) {}
    inline bool operator==(const FieldMemInfo& rhs) const
    {
      return tid == rhs.tid && fid == rhs.fid && memory == rhs.memory;
    }
    inline bool operator<(const FieldMemInfo& rhs) const
    {
      if (tid < rhs.tid)
        return true;
      else if (tid > rhs.tid)
        return false;
      if (fid < rhs.fid)
        return true;
      else if (fid > rhs.fid)
        return false;
      return memory < rhs.memory;
    }

   public:
    RegionTreeID tid;
    FieldID fid;
    Memory memory;
  };

 public:
  Legion::Mapping::LocalLock& manager_lock() { return manager_lock_; }

 private:
  Legion::Mapping::LocalLock manager_lock_{};
};

class InstanceManager : public BaseInstanceManager {
 public:
  using RegionGroupP = std::shared_ptr<RegionGroup>;

 public:
  bool find_instance(Region region,
                     FieldID field_id,
                     Memory memory,
                     Instance& result,
                     const InstanceMappingPolicy& policy = {});
  RegionGroupP find_region_group(const Region& region,
                                 const Domain& domain,
                                 FieldID field_id,
                                 Memory memory,
                                 bool exact = false);
  std::set<Instance> record_instance(RegionGroupP group,
                                     FieldID field_id,
                                     Instance instance,
                                     const InstanceMappingPolicy& policy = {});

 public:
  void erase(Instance inst);

 public:
  static InstanceManager* get_instance_manager();

 public:
  std::map<Legion::Memory, size_t> aggregate_instance_sizes() const;

 private:
  std::map<FieldMemInfo, InstanceSet> instance_sets_{};
};

class ReductionInstanceManager : public BaseInstanceManager {
 public:
  using ReductionOpID = Legion::ReductionOpID;

 public:
  bool find_instance(ReductionOpID& redop,
                     Region region,
                     FieldID field_id,
                     Memory memory,
                     Instance& result,
                     const InstanceMappingPolicy& policy = {});

  void record_instance(ReductionOpID& redop,
                       Region region,
                       FieldID field_id,
                       Instance instance,
                       const InstanceMappingPolicy& policy = {});

 public:
  static ReductionInstanceManager* get_instance_manager();

 private:
  std::map<FieldMemInfo, ReductionInstanceSet> instance_sets_{};
};

}  // namespace mapping
}  // namespace legate
