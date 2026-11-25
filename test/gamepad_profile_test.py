#!/usr/bin/env python3
"""Offline tests for BEITONG/Zikway button profiles and X-only RL entry.

No /dev/input access is required: the tests feed synthetic js events into
XboxCommandMapper, so they exercise the same mapping code as the bridge/sim.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "hardware" / "gamepad"))

from opendoge_joystick import (  # noqa: E402
    JS_EVENT_BUTTON,
    JoystickEvent,
    XboxCommandMapper,
)
from xbox_command_bridge import apply_button_profile  # noqa: E402


def make_mapper(**overrides) -> XboxCommandMapper:
    defaults = dict(
        max_vx=0.45,
        max_vy=0.25,
        max_yaw_rate=0.70,
        deadzone=0.08,
        axis_vy=0,
        axis_vx=1,
        axis_yaw=2,
        require_rb=False,
        btn_a=None,
        btn_b=None,
        btn_x=None,
        btn_y=None,
        btn_back=None,
        btn_start=None,
        btn_rb=None,
        btn_estop=None,
    )
    defaults.update(overrides)
    return XboxCommandMapper(**defaults)


def make_args(profile: str = "auto", **overrides) -> argparse.Namespace:
    values = dict(
        profile=profile,
        btn_a=None,
        btn_b=None,
        btn_x=None,
        btn_y=None,
        btn_back=None,
        btn_start=None,
        btn_rb=None,
        btn_estop=None,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def press(mapper: XboxCommandMapper, button: int) -> None:
    mapper.update([JoystickEvent(0, 1, JS_EVENT_BUTTON, button)])


def release(mapper: XboxCommandMapper, button: int) -> None:
    mapper.update([JoystickEvent(0, 0, JS_EVENT_BUTTON, button)])


def test_beitong_profile() -> None:
    mapper = make_mapper()
    apply_button_profile(
        mapper, make_args("auto"), "BEITONG BTP-KP70C BFM DONGLE")
    assert (mapper.btn_a, mapper.btn_b, mapper.btn_x, mapper.btn_y) == (0, 1, 3, 4)
    # LB/RB are 6/7; Back/Start must not inherit the Xbox 6/7 defaults.
    assert mapper.btn_rb == 7
    assert mapper.btn_back is None
    assert mapper.btn_start is None


def test_beitong_optional_m5_m6() -> None:
    # main() constructs the mapper from the CLI args, so mirror that here.
    mapper = make_mapper(btn_back=13, btn_start=14)
    args = make_args("beitong", btn_back=13, btn_start=14)
    apply_button_profile(mapper, args, "BEITONG BTP-KP70C BFM DONGLE")
    assert mapper.btn_back == 13  # M6
    assert mapper.btn_start == 14  # M5


def test_zikway_profile_unchanged() -> None:
    mapper = make_mapper()
    apply_button_profile(mapper, make_args("auto"), "Zikway HID gamepad")
    assert (mapper.btn_a, mapper.btn_b, mapper.btn_x, mapper.btn_y) == (0, 1, 3, 4)
    assert (mapper.btn_back, mapper.btn_start) == (7, 6)
    assert (mapper.btn_rb, mapper.btn_estop) == (11, 11)


def test_default_xbox_constants_unchanged() -> None:
    mapper = make_mapper()
    apply_button_profile(mapper, make_args("auto"), "Xbox 360")
    assert (mapper.btn_a, mapper.btn_b, mapper.btn_x, mapper.btn_y) == (0, 1, 2, 3)
    assert (mapper.btn_back, mapper.btn_start, mapper.btn_rb) == (6, 7, 5)


def test_x_enters_rl_without_a() -> None:
    mapper = make_mapper(btn_x=3)
    assert not mapper.active_requested and not mapper.rl_inference
    press(mapper, 3)
    assert mapper.active_requested
    assert mapper.rl_inference
    assert not mapper.position_control
    command = mapper.command()
    assert command.active and command.rl_inference


def test_b_then_x_reenters_rl() -> None:
    mapper = make_mapper(btn_b=1, btn_x=3)
    press(mapper, 1)  # B: damping
    release(mapper, 1)
    assert not mapper.active_requested and not mapper.rl_inference
    press(mapper, 3)  # X: direct RL
    assert mapper.active_requested and mapper.rl_inference


def main() -> int:
    tests = [
        test_beitong_profile,
        test_beitong_optional_m5_m6,
        test_zikway_profile_unchanged,
        test_default_xbox_constants_unchanged,
        test_x_enters_rl_without_a,
        test_b_then_x_reenters_rl,
    ]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {test.__name__}: {exc}")
    print(f"{len(tests) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
