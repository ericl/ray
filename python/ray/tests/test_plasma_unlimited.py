import numpy as np

import ray
from ray.test_utils import wait_for_condition
from ray.internal.internal_api import memory_summary

MB = 1024 * 1024


def _init_ray():
    return ray.init(
        num_cpus=2,
        object_store_memory=700e6,
        _system_config={"plasma_unlimited": 1})


def _check_spilled_mb(address, spilled=None, restored=None):
    def ok():
        s = memory_summary(address=address["redis_address"], stats_only=True)
        print(s)
        if restored:
            if "Restored {} MiB".format(restored) not in s:
                return False
        else:
            if "Restored" in s:
                return False
        if spilled:
            if "Spilled {} MiB".format(spilled) not in s:
                return False
        else:
            if "Spilled" in s:
                return False
        return True

    wait_for_condition(ok, timeout=3, retry_interval_ms=1000)


def test_fallback_when_spilling_impossible_on_put():
    try:
        address = _init_ray()
        x1 = ray.put(np.zeros(400 * MB, dtype=np.uint8))
        x1p = ray.get(x1)
        # x2 will be fallback allocated on the filesystem.
        x2 = ray.put(np.zeros(400 * MB, dtype=np.uint8))
        x2p = ray.get(x2)
        del x1p
        del x2p
        _check_spilled_mb(address, spilled=None)
    finally:
        ray.shutdown()


def test_spilling_when_possible_on_put():
    try:
        address = _init_ray()
        results = []
        for _ in range(5):
            results.append(ray.put(np.zeros(400 * MB, dtype=np.uint8)))
        _check_spilled_mb(address, spilled=1600)
    finally:
        ray.shutdown()


def test_fallback_when_spilling_impossible_on_get():
    try:
        address = _init_ray()
        x1 = ray.put(np.zeros(400 * MB, dtype=np.uint8))
        # x1 will be spilled.
        x2 = ray.put(np.zeros(400 * MB, dtype=np.uint8))
        _check_spilled_mb(address, spilled=400)
        # x1 will be restored, x2 will be spilled.
        x1p = ray.get(x1)
        _check_spilled_mb(address, spilled=800, restored=400)
        # x2 will be restored, triggering a fallback allocation.
        x2p = ray.get(x2)
        _check_spilled_mb(address, spilled=800, restored=800)
        del x1p
        del x2p
    finally:
        ray.shutdown()


def test_spilling_when_possible_on_get():
    try:
        address = _init_ray()
        x1 = ray.put(np.zeros(400 * MB, dtype=np.uint8))
        # x1 will be spilled.
        x2 = ray.put(np.zeros(400 * MB, dtype=np.uint8))
        _check_spilled_mb(address, spilled=400)
        # x1 will be restored, x2 will be spilled.
        ray.get(x1)
        _check_spilled_mb(address, spilled=800, restored=400)
        # x2 will be restored, spilling x1.
        ray.get(x2)
        _check_spilled_mb(address, spilled=800, restored=800)
    finally:
        ray.shutdown()


def test_task_unlimited():
    try:
        address = _init_ray()
        x1 = ray.put(np.zeros(400 * MB, dtype=np.uint8))
        refs = [x1]
        # x1 is spilled.
        x2 = ray.put(np.zeros(400 * MB, dtype=np.uint8))
        x2p = ray.get(x2)
        sentinel = ray.put(np.zeros(100 * MB, dtype=np.uint8))
        _check_spilled_mb(address, spilled=400)

        @ray.remote
        def consume(refs):
            # round 1: triggers fallback allocation, spilling of the sentinel
            # round 2: x2 is spilled
            ray.get(refs[0])
            # round 1: triggers fallback allocation.
            return ray.put(np.zeros(400 * MB, dtype=np.uint8))

        # round 1
        ray.get(consume.remote(refs))
        _check_spilled_mb(address, spilled=500, restored=400)

        # round 2
        del x2p
        ray.get(consume.remote(refs))
        _check_spilled_mb(address, spilled=900, restored=800)
        del sentinel
    finally:
        ray.shutdown()


if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main(["-v", __file__]))