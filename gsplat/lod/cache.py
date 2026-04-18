# SPDX-FileCopyrightText: Copyright 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GPU LRU cache of SPT cuts.

Keyed by ``spt_id``. An entry is reusable when the *current* camera distance
``d`` and the cached distance ``d_cached`` satisfy
``D_min <= d / d_cached <= D_max`` (paper §4.4). Otherwise the entry is
evicted. Write-back of modified entries is triggered on eviction; periodic
flushes (every ``flush_every`` iterations) write all dirty entries back and
clear the cache.

This class does not move data from the CPU store; the trainer / cut code
pulls params and hands them to ``insert``. The cache stores only what it
needs to rehydrate the render set: GPU-side params, hierarchy node ids, and
per-entry metadata.
"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor


@dataclass
class CacheEntry:
    """A cached SPT cut on the GPU.

    ``params`` holds gsplat-style tensors for the cut (means/scales/etc.) on
    the GPU. ``node_ids`` are hierarchy indices. ``cached_distance`` is the
    camera distance at which the cut was computed; a new lookup is allowed iff
    D_min <= new_distance / cached_distance <= D_max.
    """

    spt_id: int
    node_ids: Tensor  # [M] long, GPU
    params: Dict[str, Tensor]  # same keys as CpuGaussianStore params, on GPU
    cached_distance: float
    last_accessed: int
    dirty: bool = False

    @property
    def size(self) -> int:
        return int(self.node_ids.numel())


class GpuSptCache:
    """Write-back LRU cache of SPT cuts, bounded by a total Gaussian count.

    Attributes:
        capacity_gaussians: hard cap on total Gaussians across all entries.
        D_min, D_max: distance-ratio tolerance for reuse.
        device: GPU to cache on.

    Methods:
        lookup(spt_id, d)    -> Optional[CacheEntry]
        insert(entry)        -> None (may evict)
        invalidate(spt_ids)  -> None (e.g., after topology change)
        flush_all()          -> writes every dirty entry back and clears.
        mark_dirty(spt_id)   -> tag entry for write-back (updated params).
    """

    def __init__(
        self,
        capacity_gaussians: int,
        D_min: float = 0.8,
        D_max: float = 1.25,
        device: str | torch.device = "cuda",
    ):
        self.capacity_gaussians = int(capacity_gaussians)
        self.D_min = float(D_min)
        self.D_max = float(D_max)
        self.device = torch.device(device)
        self._entries: "OrderedDict[int, CacheEntry]" = OrderedDict()
        self._total_size: int = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    # --- Stats -----------------------------------------------------------

    @property
    def n_entries(self) -> int:
        return len(self._entries)

    @property
    def total_size(self) -> int:
        return self._total_size

    # --- Core API --------------------------------------------------------

    def lookup(self, spt_id: int, current_distance: float, iter_step: int = 0) -> Optional[CacheEntry]:
        entry = self._entries.get(spt_id)
        if entry is None:
            self.misses += 1
            return None
        ratio = current_distance / max(entry.cached_distance, 1e-12)
        if ratio < self.D_min or ratio > self.D_max:
            self.misses += 1
            return None
        entry.last_accessed = iter_step
        self._entries.move_to_end(spt_id)
        self.hits += 1
        return entry

    def insert(self, entry: CacheEntry, writeback_cb=None) -> None:
        """Insert an entry, evicting LRU entries as needed.

        Args:
            entry: the new entry (already on GPU).
            writeback_cb: optional callable(entry) invoked on each evicted
                dirty entry. If omitted, dirty entries are dropped silently —
                callers that care about preserving updates must supply one.
        """
        # If this spt_id already exists, overwrite (after accounting).
        if entry.spt_id in self._entries:
            old = self._entries.pop(entry.spt_id)
            self._total_size -= old.size

        # Evict until we fit.
        while self._total_size + entry.size > self.capacity_gaussians and self._entries:
            self._evict_one(writeback_cb)

        self._entries[entry.spt_id] = entry
        self._total_size += entry.size

    def _evict_one(self, writeback_cb=None) -> None:
        spt_id, victim = self._entries.popitem(last=False)  # LRU side
        self._total_size -= victim.size
        self.evictions += 1
        if victim.dirty and writeback_cb is not None:
            writeback_cb(victim)

    def mark_dirty(self, spt_id: int) -> None:
        e = self._entries.get(spt_id)
        if e is not None:
            e.dirty = True

    def invalidate(self, spt_ids) -> None:
        for spt_id in spt_ids:
            old = self._entries.pop(spt_id, None)
            if old is not None:
                self._total_size -= old.size

    def flush_all(self, writeback_cb=None) -> None:
        """Write every dirty entry back (if a callback is supplied) and clear.

        Paper §4.4: full flush every 1000 iters.
        """
        if writeback_cb is not None:
            for e in self._entries.values():
                if e.dirty:
                    writeback_cb(e)
        self._entries.clear()
        self._total_size = 0

    # --- Dict-like helpers for debugging -------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, spt_id: int) -> bool:
        return spt_id in self._entries

    def keys(self):
        return list(self._entries.keys())
