"""Tests for the device catalogue and device selection in lfslab.hardware."""

import sys

import pytest

from lfslab.estimator import HardwareSpec
from lfslab.hardware import CATALOGUE, pick_device


def test_catalogue_entries_are_hardware_specs():
    assert CATALOGUE
    for key, spec in CATALOGUE.items():
        assert isinstance(spec, HardwareSpec), key


@pytest.mark.parametrize("key", sorted(CATALOGUE))
def test_catalogue_values_are_positive(key):
    spec = CATALOGUE[key]
    assert spec.name
    assert spec.peak_flops_per_s > 0
    assert spec.memory_bandwidth_bytes_per_s > 0
    assert spec.device_memory_bytes > 0


def test_catalogue_names_are_unique():
    names = [spec.name for spec in CATALOGUE.values()]
    assert len(names) == len(set(names))


def test_h100_outclasses_a100():
    a100, h100 = CATALOGUE["a100_80g"], CATALOGUE["h100_80g"]
    assert h100.peak_flops_per_s > a100.peak_flops_per_s
    assert h100.memory_bandwidth_bytes_per_s > a100.memory_bandwidth_bytes_per_s


def test_pick_device_returns_known_device():
    assert pick_device() in {"cuda", "mps", "cpu"}


def test_pick_device_falls_back_to_cpu_without_torch(monkeypatch):
    # A None entry in sys.modules makes `import torch` raise ImportError.
    monkeypatch.setitem(sys.modules, "torch", None)
    assert pick_device() == "cpu"
