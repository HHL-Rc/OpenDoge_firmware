#!/usr/bin/env python3
"""Generate a MuJoCo terrain scene that includes the OpenDoge_train robot.

The training repo owns the exact XML/STLs used by the policy.  We do not copy
19 MB into sim2sim/: the scene is generated in a temporary directory, the robot
XML is copied (small), and its `assets` directory is symlinked to the training
checkout.  MuJoCo resolves `meshdir="assets"` relative to the top-level scene
directory, so this layout is required.
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    Image = None
    _PIL_ERROR = exc


def _training_xml_dir(training_root: str) -> Path:
    return Path(training_root) / "src" / "assets" / "robots" / "opendoge" / "xmls"


def _write_heightfield(path: Path, size: int, amplitude: float, seed: int) -> None:
    if Image is None:
        raise RuntimeError(
            "地形生成需要 Pillow: pip install pillow"
        ) from _PIL_ERROR
    rng = np.random.default_rng(seed)
    # Smooth low-frequency noise, tapered to zero at the patch border so the
    # robot can walk from the flat platform onto the heightfield without a step.
    coarse = rng.normal(size=(9, 9))
    x = np.linspace(0, 8, size)
    y = np.linspace(0, 8, size)
    xi, yi = np.meshgrid(x, y, indexing="ij")
    ix = np.clip(xi.astype(int), 0, 7)
    iy = np.clip(yi.astype(int), 0, 7)
    fx = xi - ix
    fy = yi - iy
    smooth = (
        coarse[ix, iy] * (1 - fx) * (1 - fy)
        + coarse[ix + 1, iy] * fx * (1 - fy)
        + coarse[ix, iy + 1] * (1 - fx) * fy
        + coarse[ix + 1, iy + 1] * fx * fy
    )
    # Taper the four edges to zero.
    edge = np.minimum(
        np.minimum(np.linspace(0, 1, size)[:, None], np.linspace(1, 0, size)[:, None]),
        np.minimum(np.linspace(0, 1, size)[None, :], np.linspace(1, 0, size)[None, :]),
    )
    taper = np.clip(edge * 5.0, 0.0, 1.0)
    height = 0.5 * (smooth - smooth.min()) / max(smooth.max() - smooth.min(), 1e-9)
    height = amplitude * np.clip(height * 0.8 + 0.2, 0.0, 1.0) * taper
    pixels = np.clip(height / max(amplitude, 1e-9) * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(pixels, mode="L").save(path)


def _scene_header(extra_asset: str = "") -> str:
    return f"""<mujoco model="opendoge_terrain">
  <include file="opendoge.xml"/>
  <statistic center="2.5 0 0.2" extent="3.0"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.25 0.25 0.25" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="grid" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="grid" texture="grid" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
    {extra_asset}
  </asset>
  <worldbody>
    <light pos="1 0 2" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="grid"/>
"""


def _build_rough(directory: Path, level: int, seed: int) -> str:
    amplitude = 0.02 + 0.015 * max(0, min(level, 3))
    _write_heightfield(directory / "assets" / "rough.png", 128, amplitude, seed)
    header = _scene_header(
        f'<hfield name="rough" file="rough.png" size="2.0 2.0 {amplitude:.4f} 0.001" '
        f'nrow="128" ncol="128"/>'
    )
    return header + f"""
    <geom name="rough" type="hfield" hfield="rough" pos="2.8 0 0" material="grid"/>
  </worldbody>
</mujoco>
"""


def _build_stairs(level: int, seed: int) -> str:
    del seed
    step_height = 0.03 + 0.01 * max(0, min(level, 3))
    step_depth = 0.30
    width = 1.2
    count = 6
    body = []
    x = 1.0
    height = 0.0
    for _ in range(count):
        height += step_height
        body.append(
            f'    <geom name="step_{x:.2f}" type="box" '
            f'size="{step_depth/2:.4f} {width/2:.4f} {height/2:.4f}" '
            f'pos="{x + step_depth/2:.4f} 0 {height/2:.4f}" material="grid"/>'
        )
        x += step_depth
    return _scene_header() + "\n".join(body) + "\n  </worldbody>\n</mujoco>\n"


def _build_boxes(level: int, seed: int) -> str:
    rng = np.random.default_rng(seed)
    max_height = 0.02 + 0.012 * max(0, min(level, 3))
    body = []
    for ix in range(10):
        for iy in range(5):
            x = 1.0 + ix * 0.42
            y = -0.9 + iy * 0.45
            height = float(rng.uniform(0.5, 1.0) * max_height)
            body.append(
                f'    <geom name="box_{ix}_{iy}" type="box" '
                f'size="0.20 0.20 {height/2:.4f}" '
                f'pos="{x:.4f} {y:.4f} {height/2:.4f}" material="grid"/>'
            )
    return _scene_header() + "\n".join(body) + "\n  </worldbody>\n</mujoco>\n"


def _build_slope(level: int) -> str:
    angle = math.radians(5.0 + 4.0 * max(0, min(level, 3)))
    half_length = 1.5
    half_width = 0.8
    half_thickness = 0.05
    cx = 1.0 + half_length * math.cos(angle)
    cz = half_length * math.sin(angle)
    quat = (math.cos(angle / 2), 0.0, math.sin(angle / 2), 0.0)
    return _scene_header() + f"""
    <geom name="ramp" type="box" size="{half_length} {half_width} {half_thickness}"
          pos="{cx:.4f} 0 {cz:.4f}" quat="{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}"
          material="grid"/>
  </worldbody>
</mujoco>
"""


def build_scene(
    training_root: str,
    terrain: str,
    level: int = 1,
    seed: int = 0,
) -> str:
    """Create a terrain scene and return its XML path.

    The returned path lives in a temp dir whose `opendoge.xml` is a copy of the
    training robot XML and whose `assets` is a symlink to the training assets.
    The temp dir intentionally lives for the process lifetime.
    """
    xml_dir = _training_xml_dir(training_root)
    robot_xml = xml_dir / "opendoge.xml"
    assets = xml_dir / "assets"
    if not robot_xml.is_file() or not assets.is_dir():
        raise FileNotFoundError(
            f"OpenDoge_train assets not found under {xml_dir}"
        )

    directory = Path(tempfile.mkdtemp(prefix="opendoge_terrain_"))
    shutil.copyfile(robot_xml, directory / "opendoge.xml")
    # Create a local assets dir with symlinks to the training STLs. The hfield
    # image also lives here because MuJoCo resolves it through the asset dir.
    local_assets = directory / "assets"
    local_assets.mkdir()
    for item in assets.iterdir():
        if item.is_file():
            os.symlink(item, local_assets / item.name)

    if terrain == "rough":
        xml = _build_rough(directory, level, seed)
    elif terrain == "stairs":
        xml = _build_stairs(level, seed)
    elif terrain == "boxes":
        xml = _build_boxes(level, seed)
    elif terrain == "slope":
        xml = _build_slope(level)
    else:
        raise ValueError(f"unsupported terrain: {terrain}")

    scene = directory / "scene.xml"
    scene.write_text(xml, encoding="utf-8")
    print(f"[TERRAIN] {terrain} level={level} seed={seed} -> {scene}")
    return str(scene)
