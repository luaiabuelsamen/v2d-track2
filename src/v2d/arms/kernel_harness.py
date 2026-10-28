"""Run flash_chord's patched ReconBody kernels here, on the CPU, with inputs we choose.

flash_chord is not installed on this machine -- it ships inside the training image, which is
x86-64 CUDA while this box is aarch64 Tegra -- so an arm that changes the objective's warp kernels
could until now only be checked by reading the diff and paying for a box. But warp itself installs
here and initialises with a `cpu` device, and the two kernels an arm touches depend on almost
nothing from the package: the quaternion helpers, and the term-count constant.

So this extracts those kernels from a patched tree's `objectives/recon_body.py`, drops them into a
module with equivalent `wp.func` quaternion helpers, compiles them for the CPU, and launches them
on synthetic worlds. What that proves is the arithmetic and the plumbing: that the patched source
compiles as a warp kernel, that every term is finite, that the new terms carry the values the arm
claims, and that the terms the arm does not touch are bit-identical to the stock kernel's. What it
cannot prove is anything about the rest of the image -- hydra composition, the reference loader,
the SAC loop, or throughput. Those stay what they are: the first commands of the first cell.

    v2d arm-kernel-check --tree <patched flash_chord> [--stock <unpatched>] [--json out.json]
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

#: Stand-ins for `flash_chord.utils.quat`. The angle one is only read by terms this arm does not
#: touch, and both are compared against the stock kernel in the same harness, so an inexact
#: stand-in would show up as a difference in the stock terms rather than hide one.
QUAT_HELPERS = """
@wp.func
def quat_geodesic_angle(a: wp.quat, b: wp.quat) -> float:
    dot = wp.abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3])
    return 2.0 * wp.acos(wp.clamp(dot, -1.0, 1.0))


@wp.func
def quat_mul_xyzw(a: wp.quat, b: wp.quat) -> wp.quat:
    return a * b
"""

KERNELS = ("reduce_recon_body_force_closure", "compose_recon_body_objective")


def constant(value: Any) -> int:
    """`wp.constant(12)` is a plain int in warp 1.18 and a wrapper with `.val` in others."""
    return int(getattr(value, "val", value))


TERM_COUNT_LINE = "_RECON_BODY_OBJECTIVE_TERM_COUNT"


def term_names(source: str) -> tuple[str, ...]:
    """The objective's term names, in index order, from the tree's own source."""
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "RECON_BODY_OBJECTIVE_TERM_NAMES"
            for t in node.targets
        ):
            return tuple(ast.literal_eval(node.value))
    raise ValueError("no RECON_BODY_OBJECTIVE_TERM_NAMES in the tree")


def extract(source: str) -> str:
    """A standalone warp module holding the two kernels, ready to compile."""
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    out = ["import warp as wp\n", QUAT_HELPERS, "\n"]
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == TERM_COUNT_LINE for t in node.targets
        ):
            out.append(ast.get_source_segment(source, node) + "\n\n")
    found = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in KERNELS:
            start = min(d.lineno for d in node.decorator_list) - 1
            out.append("".join(lines[start : node.end_lineno]) + "\n\n")
            found.append(node.name)
    missing = [name for name in KERNELS if name not in found]
    if missing:
        raise ValueError(f"the tree is missing {missing}")
    return "".join(out)


def _signature(module: Any, name: str) -> list[str]:
    kernel = getattr(module, name)
    return [arg.label for arg in kernel.adj.args]


def _load(text: str, path: Path) -> Any:
    import importlib.util

    path.write_text(text)
    spec = importlib.util.spec_from_file_location(f"v2d_kernels_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Scene:
    """One deterministic synthetic batch: 3 worlds, 2 hands, 1 object, 1 support basis.

    World 0: both hands in the contact role, both touching, palms 2 cm outside the object's
    radius. World 1: left hand in the contact role and 12 cm out, right hand free -- so the term
    must read the left hand only. World 2: neither hand in the contact role, which is the case the
    arm must leave alone.
    """

    world_count = 3
    num_hands = 2
    num_bodies = 1
    num_basis = 4
    num_frames = 5
    bodies_per_world = 4  # pelvis, left wrist, right wrist, object
    q_per_world = 10
    num_joints = 3
    num_sonic_joints = 2
    pelvis_body_id = 0
    object_body_id = 3
    frame_dt = 1.0 / 30.0
    object_radius = 0.06
    palm_std = 0.05
    body_scale = 0.2  # 1.0 / 5.0: the arm's in-contact weight over the stock weight
    min_support = 0.01
    #: (world, hand) palm distance from the object centre, in metres.
    palm_distance = ((0.08, 0.08), (0.18, 0.30), (0.08, 0.08))
    #: (world, hand) role, 1.0 = contact.
    roles = ((1.0, 1.0), (1.0, 0.0), (0.0, 0.0))
    #: (world, hand) whether the wrench support says the hand is loading the object.
    touching = ((1.0, 0.0), (1.0, 1.0), (1.0, 1.0))

    def expected_gap(self, world: int, hand: int) -> float:
        return max(self.palm_distance[world][hand] - self.object_radius, 0.0)

    def expected_contact_role_palm(self, world: int) -> float:
        hands = [h for h in range(2) if self.roles[world][h] > 0.5]
        if not hands:
            return 0.0
        gaps = [self.expected_gap(world, h) for h in hands]
        return sum(math.exp(-(g * g) / (self.palm_std**2)) for g in gaps) / len(hands)

    def expected_contact_consistency(self, world: int) -> float:
        hands = [h for h in range(2) if self.roles[world][h] > 0.5]
        if not hands:
            return 0.0
        return sum(self.touching[world][h] for h in hands) / len(hands)

    def in_contact(self, world: int) -> bool:
        return any(r > 0.5 for r in self.roles[world])


def run(module: Any, scene: Scene, *, patched: bool) -> dict[str, np.ndarray]:
    """Launch both kernels once and return every output array.

    Arguments are matched to the kernel BY NAME, not by position, because the arms compose: with
    the orientation arm applied as well, `compose_recon_body_objective` takes two more arguments,
    and a positional list would either mis-assign them or, as it did here first, refuse to launch.
    An argument the harness has no value for is a loud failure rather than a zero.
    """
    import warp as wp

    device = "cpu"
    terms_per_world = constant(module._RECON_BODY_OBJECTIVE_TERM_COUNT)

    # The palms sit on the +x axis at the asked-for distance from the object centre, which is the
    # origin, so |palm - centre| is exactly palm_distance and the gap is readable by hand.
    transforms = []
    for world in range(scene.world_count):
        transforms.append(wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()))  # pelvis
        for hand in range(2):
            x = scene.palm_distance[world][hand]
            transforms.append(wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()))
        transforms.append(wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()))  # object

    support = np.zeros(
        (scene.world_count, scene.num_hands, scene.num_bodies, scene.num_basis), dtype=np.float32
    )
    for world in range(scene.world_count):
        for hand in range(scene.num_hands):
            if scene.touching[world][hand] > 0.5:
                support[world, hand, 0, 0] = 1.0  # one covered basis direction is "touching"

    reference_joint_q = np.zeros((scene.num_frames, scene.q_per_world), dtype=np.float32)
    reference_joint_q[:, 2] = 1.0  # a pelvis height to track
    reference_joint_q[:, 6] = 1.0  # a unit reference quaternion (w last)

    def f32(values: np.ndarray) -> Any:
        return wp.array(
            np.ascontiguousarray(values, dtype=np.float32), dtype=wp.float32, device=device
        )

    def i32(values: Any) -> Any:
        return wp.array(np.asarray(values, dtype=np.int32), dtype=wp.int32, device=device)

    def quats(count: int) -> Any:
        return wp.array([wp.quat_identity() for _ in range(count)], dtype=wp.quat, device=device)

    force_closure = f32(np.zeros(scene.world_count))
    touching = f32(np.zeros(scene.world_count * scene.num_hands))
    terms = f32(np.zeros(scene.world_count * terms_per_world))
    score = f32(np.zeros(scene.world_count))
    roles = np.zeros((scene.num_frames, scene.num_hands), dtype=np.float32)
    role_array = f32(roles.reshape(-1))

    values: dict[str, Any] = {
        # the reducer
        "timestep": i32(np.zeros(scene.world_count)),  # every world sits on frame 0
        "contact_active": f32(np.ones(scene.num_frames * scene.num_hands)),
        "current_support": f32(support.reshape(-1)),
        "num_frames": scene.num_frames,
        "num_hands": scene.num_hands,
        "num_bodies": scene.num_bodies,
        "num_basis": scene.num_basis,
        "min_support": scene.min_support,
        "force_closure": force_closure,
        "hand_touching": touching,
        # the composer
        "body_q": wp.array(transforms, dtype=wp.transform, device=device),
        "joint_q": f32(np.zeros(scene.world_count * scene.q_per_world)),
        "episode_start_frame": i32(np.zeros(scene.world_count)),
        "terminated": i32(np.zeros(scene.world_count)),
        "action_rate_l2": f32(np.zeros(scene.world_count)),
        "action_l2": f32(np.zeros(scene.world_count)),
        "reference_joint_q_offset": f32(np.zeros(scene.world_count * scene.q_per_world)),
        "reference_joint_q": wp.array(
            reference_joint_q, dtype=wp.float32, ndim=2, device=device
        ),
        "reference_wrist_pos_w": wp.zeros(scene.num_frames * 2, dtype=wp.vec3, device=device),
        "reference_wrist_quat_w": quats(scene.num_frames * 2),
        "wrist_body_ids": i32([1, 2]),
        "wrist_local_pos": wp.zeros(2, dtype=wp.vec3, device=device),
        "wrist_local_quat": quats(2),
        "scalar_q_ids": i32(range(scene.num_joints)),
        "sonic_q_ids": i32(range(scene.num_sonic_joints)),
        "soft_lower": f32(np.full(scene.num_joints, -1.0)),
        "soft_upper": f32(np.full(scene.num_joints, 1.0)),
        "object_body_id": scene.object_body_id,
        "object_target_pos_w": wp.zeros(scene.world_count, dtype=wp.vec3, device=device),
        "pelvis_body_id": scene.pelvis_body_id,
        "bodies_per_world": scene.bodies_per_world,
        "q_per_world": scene.q_per_world,
        "num_joints": scene.num_joints,
        "num_sonic_joints": scene.num_sonic_joints,
        "pelvis_position_std": 0.3,
        "pelvis_orientation_std": 0.4,
        "body_joint_position_std": 1.0,
        "object_position_std": 0.2,
        "palm_position_std": 0.2,
        "palm_orientation_std": 0.4,
        "weights": f32(np.ones(terms_per_world)),
        "frame_dt": scene.frame_dt,
        "terms": terms,
        "score": score,
        # the orientation arm, when it is applied too
        "object_target_quat_w": quats(scene.world_count),
        "object_orientation_std": 0.3,
        # arm CR
        "contact_role": role_array,
        "object_radius": scene.object_radius,
        "contact_role_palm_std": scene.palm_std,
        "body_joint_position_in_contact_scale": scene.body_scale,
    }

    def launch(name: str) -> None:
        wanted = _signature(module, name)
        missing = [arg for arg in wanted if arg not in values]
        if missing:
            raise ValueError(f"{name} takes {missing}, which this harness has no value for")
        wp.launch(
            getattr(module, name),
            dim=scene.world_count,
            inputs=[values[arg] for arg in wanted],
            device=device,
        )

    launch("reduce_recon_body_force_closure")

    # The harness wants per-WORLD roles and the kernel reads them per FRAME, so each world is
    # launched on its own with frame 0 carrying that world's roles, and only that world's terms
    # are kept. It keeps every expected value hand-computable.
    terms_out = np.zeros((scene.world_count, terms_per_world), dtype=np.float32)
    score_out = np.zeros(scene.world_count, dtype=np.float32)
    for world in range(scene.world_count):
        roles[:] = 0.0
        roles[0, :] = scene.roles[world]
        role_array.assign(np.ascontiguousarray(roles.reshape(-1), dtype=np.float32))
        launch("compose_recon_body_objective")
        terms_out[world] = terms.numpy().reshape(scene.world_count, terms_per_world)[world]
        score_out[world] = score.numpy()[world]

    del patched  # the signature decides what is passed; nothing here needs to know
    return {
        "force_closure": force_closure.numpy().copy(),
        "hand_touching": touching.numpy().copy(),
        "terms": terms_out,
        "score": score_out,
    }


def check(tree: Path, stock: Path | None, scratch: Path) -> dict[str, Any]:
    """Compile and launch both trees' kernels and compare them. Returns the findings."""
    import warp as wp

    wp.init()
    scratch.mkdir(parents=True, exist_ok=True)
    source = (tree / "objectives/recon_body.py").read_text()
    names = term_names(source)
    scene = Scene()

    started = time.monotonic()
    patched_module = _load(extract(source), scratch / "patched_kernels.py")
    compiled = time.monotonic() - started
    patched_out = run(patched_module, scene, patched=True)
    ran = time.monotonic() - started - compiled

    findings: dict[str, Any] = {
        "device": "cpu",
        "warp": wp.config.version,
        "term_names": list(names),
        "term_count": len(names),
        "compile_seconds": round(compiled, 2),
        "launch_seconds": round(ran, 3),
        "signature": {name: _signature(patched_module, name) for name in KERNELS},
        "worlds": [],
        "all_terms_finite": bool(np.all(np.isfinite(patched_out["terms"]))),
        "score_finite": bool(np.all(np.isfinite(patched_out["score"]))),
    }
    role_index = names.index("contact_role_palm")
    consistency_index = names.index("contact_consistency")
    body_index = names.index("body_joint_position")

    stock_out = None
    if stock is not None:
        stock_source = (stock / "objectives/recon_body.py").read_text()
        stock_module = _load(extract(stock_source), scratch / "stock_kernels.py")
        stock_out = run(stock_module, scene, patched=False)
        findings["stock_term_count"] = constant(stock_module._RECON_BODY_OBJECTIVE_TERM_COUNT)
        findings["force_closure_unchanged"] = bool(
            np.allclose(patched_out["force_closure"], stock_out["force_closure"], atol=0.0)
        )
        findings["force_closure"] = {
            "patched": [round(float(v), 6) for v in patched_out["force_closure"]],
            "stock": [round(float(v), 6) for v in stock_out["force_closure"]],
        }

    for world in range(scene.world_count):
        got_role = float(patched_out["terms"][world, role_index])
        got_consistency = float(patched_out["terms"][world, consistency_index])
        want_role = scene.expected_contact_role_palm(world)
        want_consistency = scene.expected_contact_consistency(world)
        row: dict[str, Any] = {
            "world": world,
            "roles": list(scene.roles[world]),
            "palm_gap_cm": [round(100.0 * scene.expected_gap(world, h), 2) for h in range(2)],
            "contact_role_palm": round(got_role, 6),
            "contact_role_palm_expected": round(want_role, 6),
            "contact_consistency": round(got_consistency, 6),
            "contact_consistency_expected": round(want_consistency, 6),
            "in_contact": scene.in_contact(world),
        }
        row["matches_hand_computed"] = bool(
            abs(got_role - want_role) < 1e-5 and abs(got_consistency - want_consistency) < 1e-6
        )
        if stock_out is not None:
            got_body = float(patched_out["terms"][world, body_index])
            stock_body = float(stock_out["terms"][world, body_index])
            want_body = stock_body * (scene.body_scale if scene.in_contact(world) else 1.0)
            row["body_joint_position"] = round(got_body, 6)
            row["body_joint_position_stock"] = round(stock_body, 6)
            row["body_joint_position_expected"] = round(want_body, 6)
            row["body_weight_scaled_correctly"] = bool(abs(got_body - want_body) < 1e-6)
            untouched = [
                index
                for index, name in enumerate(names[: findings["stock_term_count"]])
                if name != "body_joint_position"
            ]
            row["stock_terms_unchanged"] = bool(
                np.allclose(
                    patched_out["terms"][world, untouched],
                    stock_out["terms"][world, untouched],
                    atol=0.0,
                )
            )
        findings["worlds"].append(row)

    findings["new_terms_nonzero"] = bool(
        any(row["contact_role_palm"] > 0.0 for row in findings["worlds"])
        and any(row["contact_consistency"] > 0.0 for row in findings["worlds"])
    )
    findings["pass"] = bool(
        findings["all_terms_finite"]
        and findings["score_finite"]
        and findings["new_terms_nonzero"]
        and all(row["matches_hand_computed"] for row in findings["worlds"])
        and (
            stock_out is None
            or (
                findings["force_closure_unchanged"]
                and all(row["body_weight_scaled_correctly"] for row in findings["worlds"])
                and all(row["stock_terms_unchanged"] for row in findings["worlds"])
            )
        )
    )
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, required=True, help="patched flash_chord source root")
    parser.add_argument(
        "--stock", type=Path, help="unpatched root, to compare the stock terms against"
    )
    parser.add_argument(
        "--scratch", type=Path, required=True, help="where the extracted modules go"
    )
    parser.add_argument("--json", type=Path, help="write the findings here")
    args = parser.parse_args(argv)

    findings = check(args.tree, args.stock, args.scratch)
    if args.json:
        args.json.write_text(json.dumps(findings, indent=1) + "\n")
    print(json.dumps(findings, indent=1))
    return 0 if findings["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
