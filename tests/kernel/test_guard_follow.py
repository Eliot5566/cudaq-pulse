# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A cache guard follows a helper into the globals it reads however the kernel
reaches it: as a function, a bound method, a classmethod, a functools.partial,
or an item of a dict, list or tuple."""

import functools
import importlib

import cudaq_pulse as pulse
import pytest

from tests._handles import Q

_guard = importlib.import_module("cudaq_pulse.kernel._guard")

AMP = 0.1


def _helper(scale=1.0):
    return AMP * scale


class _Gates:
    def x(self):
        return AMP

    @classmethod
    def y(cls):
        return AMP


GATES = _Gates()
TABLE = {"x": _helper}
SEQUENCE = [_helper]
PARTIAL = functools.partial(_helper, scale=2.0)


def _via_method():
    return GATES.x()


def _via_classmethod():
    return _Gates.y()


def _via_dict():
    return TABLE["x"]()


def _via_list():
    return SEQUENCE[0]()


def _via_partial():
    return PARTIAL()


@pytest.fixture
def amp():
    def set_amp(value):
        globals()["AMP"] = value

    yield set_amp
    globals()["AMP"] = 0.1


@pytest.mark.parametrize("fn", [_via_method, _via_classmethod, _via_dict, _via_list, _via_partial])
def test_a_global_a_reached_helper_reads_is_guarded(fn, amp):
    guard = _guard.build_guard(fn)
    assert guard.check()
    amp(0.2)
    assert not guard.check()


class _Pulses:
    def x(self, d):
        drive(d, square(16, AMP))  # noqa: F821


PULSES = _Pulses()


@pulse.kernel
def _drive_x(q):
    PULSES.x(get_drive_line(q))  # noqa: F821


def test_a_kernel_compiles_again_when_a_method_helper_global_changes(amp):
    _drive_x.lower(Q[0])
    amp(0.2)
    assert "2.000000e-01" in _drive_x.lower(Q[0]).mlir()
    assert _drive_x.cache_info().compiled == 2
