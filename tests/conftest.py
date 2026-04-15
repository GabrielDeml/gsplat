# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Pytest configuration and shared fixtures for gsplat tests.

This file is automatically discovered by pytest and applies to all test files
in this directory and subdirectories.

Device parametrization
----------------------
Tests that should run across backends can request the ``device`` fixture. It
is parametrized over ``cuda``, ``mps``, and ``cpu``; parameters whose backend
is unavailable on the host are skipped automatically, so a single test body
covers all platforms without per-test ``skipif`` boilerplate.

Example::

    def test_zeros(device):
        x = torch.zeros(3, device=device)
        assert x.device.type in {"cuda", "mps", "cpu"}
"""

import gc
import os

import pytest
import torch
import torch.distributed


def _gpu_available():
    """Check if any GPU backend (CUDA or MPS) is available."""
    return torch.cuda.is_available() or (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )


def _get_device():
    """Return the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.fixture(params=["cuda", "mps", "cpu"])
def device(request):
    """Parametrized device fixture covering cuda, mps, and cpu.

    Unavailable backends are skipped per-parameter so tests that request this
    fixture run once per backend present on the host.
    """
    backend = request.param
    if backend == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if backend == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        pytest.skip("MPS not available")
    return torch.device("cuda:0" if backend == "cuda" else backend)


@pytest.fixture(autouse=True)
def setup_test_environment():
    """
    Autouse fixture that runs before every test to ensure:
    1. Deterministic random seed
    2. GPU cache is cleared
    3. Garbage collection is performed

    This fixture automatically applies to all tests in this directory
    without needing to be explicitly requested.
    """

    seed = 42

    # Set seed based on test name for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()

    # Run garbage collection
    gc.collect()

    # Yield to run the test
    yield

    # Optional: cleanup after test
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


@pytest.fixture(scope="session")
def dist_init():
    """Initialize a single-process distributed group for testing distributed code paths.

    With world_size=1 the all-gather / all-to-all ops become identity operations,
    but the code path inside ``rasterization(distributed=True)`` is still exercised.
    """
    if not torch.cuda.is_available():
        yield
        return

    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        torch.distributed.init_process_group(backend="nccl", world_size=1, rank=0)
        # Warm up the communicator required by batch_isend_irecv.
        _ = [None]
        torch.distributed.all_gather_object(_, 0)

    yield

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
