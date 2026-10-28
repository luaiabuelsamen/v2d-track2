#!/usr/bin/env python3
"""Arm CR: a contact-role reward for flash_chord's ReconBody objective, DONIC-style.

NVIDIA's objective rewards both palms at all times -- `palm_position` and `palm_orientation` are
summed over both hands on every frame, whether or not that hand has anything to do. A hand the
reference leaves idle is therefore scored exactly as hard as the hand that is carrying the object,
and the policy spends capacity putting an empty hand in the right place. Our own measurements say
the carrying hand is where the score is: of the nine clips under 0.30 AUC, seven lose the object
mid-clip while the body keeps tracking, and `docs/FAILURE_MOMENTS.md` names a grip that opens or
slides as the mechanism in five of them.

This adds two terms, both gated on a per-hand ROLE read from the reference:

- `contact_role_palm` (weight 2.0, std 0.05 m): for each hand whose role on this frame is
  *contact*, `exp(-gap^2 / std^2)` where `gap` is the palm-to-object-surface distance, clamped at
  zero. Averaged over the contact-role hands, so the term stays in [0, 1] like every other shaped
  term in this objective -- the weights are calibrated against that range, and a sum would make a
  two-handed clip's term twice a one-handed clip's for the same quality of grasp. A hand in the
  *free* role contributes nothing, which is the whole point of the arm.
- `contact_consistency` (weight 1.0): of those same contact-role hands, the fraction the simulator
  reports actually loading the object. This is the sim's report, not the reference's label: it is
  taken from the wrench support the force-closure term already computes, so a hand can be in the
  right place (`contact_role_palm` high) and still score zero here if it is not touching anything.

and it scales one existing term: on frames where either hand is in the contact role,
`body_joint_position` is multiplied by `--body-weight-in-contact / body_joint_position_weight`, so
the effective weight on whole-body joint tracking falls from the stock 5.0 to 1.0 exactly while a
hand is supposed to be holding something. The reward is a weighted sum, so scaling the term is the
only way to make a weight frame-dependent -- `set_weights` is per stage, not per frame.

**The surface distance is a sphere approximation**, and deliberately the same one NVIDIA already
uses: `|palm - object_centre| - object_radius`, with `object_radius` read once at build time from
the `CommandBuffers` field the wrench-support kernel is launched with. It is exact for the compact
objects and conservative for the long ones -- the cane and the squeegee are 0.6 m long, so a grip
on the handle reads as ~25 cm of gap and the term saturates low. If the arm wins, the honest next
step is a per-object point cloud in the object frame (the kit's own vertex sampler, which is what
`docs/METRICS_DEEP_DIVE.md` measures palm-to-surface with); that is a bigger change to the kernel's
inputs, and it is not what this arm is testing.

Roles come from `--roles`/`V2D_CONTACT_ROLES`, an `.npz` per clip holding `role` (or `roles`, or
`contact_role`) of shape (frames, 2) -- side order `("left", "right")`, nonzero meaning contact.
A (2, frames) array is transposed, because that is the layout `hand_contact_active` ships in. Until
interp's `results/contact_roles/eXX.npz` exist the roles are **derived from the reference's own
contact labels**, which is the same array the force-closure term is gated on: a defensible stub,
and the arm is then "reward the hand the reference says is holding it". If `V2D_CONTACT_ROLES` is
set but cannot be read, the run dies rather than falling back to the stub -- an arm that silently
trains the stub under the real arm's name is the failure this whole patch style exists to prevent.

Term indices are computed from the installed source rather than hardcoded, so this composes with
`patch_object_orientation.py` in either order. Run inside the training container, before training:

    python patch_contact_role.py /workspace/src/flash_chord [--check] [--weight 2.0] [--std 0.05]
        [--consistency-weight 1.0] [--body-weight-in-contact 1.0]

Idempotent; refuses, having changed nothing, if a value differs from one the tree already carries,
if any anchor is missing or ambiguous, or if the patched Python does not parse.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

MARKER = "contact_role_palm"
TERM_NAMES = ("contact_role_palm", "contact_consistency")

DEFAULT_WEIGHT = 2.0
DEFAULT_STD = 0.05
DEFAULT_CONSISTENCY_WEIGHT = 1.0
DEFAULT_BODY_WEIGHT_IN_CONTACT = 1.0

TERM_COUNT = re.compile(r"^_RECON_BODY_OBJECTIVE_TERM_COUNT = wp\.constant\((\d+)\)$", re.MULTILINE)


def num(value: float) -> str:
    """A float that still looks like one: `2.0`, not `2`.

    The same text goes into a `float` dataclass default and into the objective YAML, and YAML
    would read a bare `2` as an int and hand hydra an int where the config declares a float.
    """
    text = f"{value:g}"
    return text if ("." in text or "e" in text) else f"{text}.0"

#: The objective-yaml keys this arm owns, with the CLI option each one comes from. Read back on a
#: second apply so a tree patched at one setting cannot be re-reported as another.
YAML_KEYS = {
    "contact_role_palm_weight": "--weight",
    "contact_role_palm_std": "--std",
    "contact_consistency_weight": "--consistency-weight",
    "body_joint_position_weight_in_contact": "--body-weight-in-contact",
}


class Refused(SystemExit):
    """Nothing has been written. The message says which anchor or value stopped it."""


def _once(text: str, anchor: str, where: str) -> None:
    count = text.count(anchor)
    if count != 1:
        raise Refused(
            f"FATAL: {where}: anchor occurs {count} times, expected exactly 1.\n"
            f"  anchor: {anchor!r}\n"
            "  flash_chord has changed, or another patch moved it; re-derive this patch against "
            "the installed source rather than letting it apply somewhere unintended."
        )


def replace(text: str, old: str, new: str, where: str) -> str:
    """Exact single-occurrence replacement."""
    _once(text, old, where)
    return text.replace(old, new, 1)


def insert_before_close(text: str, region: str, close: str, addition: str, where: str) -> str:
    """Insert `addition` just before the first `close` after `region`.

    Anchoring on a region's opening line and its closing bracket, rather than on the last item in
    it, is what lets this patch and `patch_object_orientation.py` both append to the same tuples in
    either order: neither one's anchor is the other's inserted text.
    """
    _once(text, region, where)
    start = text.index(region) + len(region)
    end = text.find(close, start)
    if end < 0:
        raise Refused(
            f"FATAL: {where}: found the region {region.strip()!r} but not its closing {close!r}; "
            "the file is not shaped the way this patch expects."
        )
    return text[:end] + addition + text[end:]


def term_count(text: str, where: str) -> int:
    found = TERM_COUNT.search(text)
    if found is None:
        raise Refused(
            f"FATAL: {where}: no _RECON_BODY_OBJECTIVE_TERM_COUNT line; cannot tell which term "
            "indices are free."
        )
    return int(found.group(1))


ROLE_HELPER = '''
def _v2d_contact_roles(labels, num_frames: int, num_hands: int):
    """Per-hand contact ROLE as (frames, hands) float32 in {0, 1}. Our addition, arm CR.

    From $V2D_CONTACT_ROLES -- a .npz, or a directory plus $V2D_CLIP_TAG -- holding `role`,
    `roles` or `contact_role` of shape (frames, hands); a (hands, frames) array is transposed,
    since that is how hand_contact_active ships. With the variable unset the reference's own
    contact labels are the roles, which is the documented stub. Set but unreadable is fatal: a
    silent fall back to the stub would put the stub on the record under the arm's name.
    """
    import os

    source = os.environ.get("V2D_CONTACT_ROLES", "").strip()
    if not source:
        return labels.copy()
    path = source
    if os.path.isdir(path):
        tag = os.environ.get("V2D_CLIP_TAG", "").strip()
        if not tag:
            raise ValueError(
                f"V2D_CONTACT_ROLES={source!r} is a directory, so V2D_CLIP_TAG must name the clip"
            )
        path = os.path.join(path, f"{tag}.npz")
    if not os.path.isfile(path):
        raise ValueError(f"V2D_CONTACT_ROLES: {path} does not exist")
    with np.load(path) as data:
        keys = list(data)
        for key in ("role", "roles", "contact_role"):
            if key in keys:
                roles = np.asarray(data[key], dtype=np.float32)
                break
        else:
            raise ValueError(f"V2D_CONTACT_ROLES: {path} has keys {keys}, none of role/roles/contact_role")
    if roles.ndim != 2:
        raise ValueError(f"V2D_CONTACT_ROLES: {path} role array has shape {roles.shape}, expected 2-D")
    if roles.shape == (num_hands, num_frames) and num_frames != num_hands:
        roles = roles.T
    if roles.shape != (num_frames, num_hands):
        raise ValueError(
            f"V2D_CONTACT_ROLES: {path} role array has shape {roles.shape}, "
            f"expected {(num_frames, num_hands)}"
        )
    if not np.all(np.isfinite(roles)):
        raise ValueError(f"V2D_CONTACT_ROLES: {path} role array is not finite")
    return np.ascontiguousarray(roles > 0.5, dtype=np.float32)


'''


def objective_edits(
    text: str,
    weight: float,
    std: float,
    consistency_weight: float,
    body_weight_in_contact: float,
    where: str,
) -> str:
    """Patch `objectives/recon_body.py`, in memory. Raises Refused without writing anything."""
    count = term_count(text, where)
    first, second = count, count + 1

    # (1) the two term names and the count, in one regex so they cannot disagree. Appended, so
    # every existing index -- and every weight the curriculum maps onto it -- is untouched.
    names_and_count = re.compile(
        r'(    "force_closure",\n(?:    "[a-z_]+",\n)*\)\n)_RECON_BODY_OBJECTIVE_TERM_COUNT = '
        r"wp\.constant\(\d+\)"
    )
    found = names_and_count.search(text)
    if found is None:
        raise Refused(
            f"FATAL: {where}: could not find the term-name tuple ending in force_closure followed "
            "by the term count; flash_chord has changed."
        )
    added = "".join(f'    "{name}",\n' for name in TERM_NAMES)
    block = found.group(1)[: -len(")\n")] + added + ")\n"
    text = (
        text[: found.start()]
        + block
        + f"_RECON_BODY_OBJECTIVE_TERM_COUNT = wp.constant({count + 2})"
        + text[found.end() :]
    )

    # (2) the config: two weights, one std, and the in-contact body weight. Anchored on `sides`,
    # the one field no other arm's patch touches.
    text = replace(
        text,
        '    sides: tuple[str, ...] = ("left", "right")\n',
        f"    contact_role_palm_weight: float = {num(weight)}\n"
        f"    contact_consistency_weight: float = {num(consistency_weight)}\n"
        f"    contact_role_palm_std: float = {num(std)}\n"
        f"    body_joint_position_weight_in_contact: float = {num(body_weight_in_contact)}\n"
        '    sides: tuple[str, ...] = ("left", "right")\n',
        where,
    )

    # (3) validation. The std joins the positive-shaping check; the in-contact weight only has to
    # be finite, which the `values` tuple covers, because 0.0 is a legitimate setting for it.
    text = insert_before_close(
        text,
        "        values = self.weights + (\n",
        "        )\n",
        "            self.body_joint_position_weight_in_contact,\n",
        f"{where} (__post_init__ values)",
    )
    text = insert_before_close(
        text,
        "            for value in (\n",
        "            )\n",
        "                self.contact_role_palm_std,\n",
        f"{where} (__post_init__ positives)",
    )

    # (4) the weights tuple, in term-name order, so the new indices get the new weights.
    text = insert_before_close(
        text,
        "    def weights(self) -> tuple[float, ...]:\n        return (\n",
        "        )\n",
        "            self.contact_role_palm_weight,\n            self.contact_consistency_weight,\n",
        f"{where} (weights property)",
    )

    # (5) the role loader, next to the kernels that use it.
    text = replace(
        text,
        "@wp.kernel\ndef reduce_recon_body_force_closure(",
        ROLE_HELPER.lstrip("\n") + "@wp.kernel\ndef reduce_recon_body_force_closure(",
        where,
    )

    # (6) the force-closure reducer also reports, per hand, whether the SIM says that hand is
    # loading the object. The force_closure value itself is unchanged: `covered` is now computed
    # for every hand instead of only the labelled ones, but it is still only added to `reward`
    # (and `active_hands`) for a hand the label gates in.
    text = replace(
        text,
        "    min_support: float,\n    force_closure: wp.array(dtype=wp.float32),\n) -> None:",
        "    min_support: float,\n    force_closure: wp.array(dtype=wp.float32),\n"
        "    hand_touching: wp.array(dtype=wp.float32),\n) -> None:",
        where,
    )
    text = replace(
        text,
        """    for hand in range(num_hands):
        if contact_active[frame * num_hands + hand] > 0.5:
            active_hands += 1.0
            covered = float(0.0)  # noqa: UP018
            for basis in range(num_basis):
                best = float(0.0)  # noqa: UP018
                for body in range(num_bodies):
                    index = ((world * num_hands + hand) * num_bodies + body) * num_basis + basis
                    best = wp.max(best, current_support[index])
                if best > min_support:
                    covered += 1.0
            reward += covered / float(num_basis)
""",
        """    for hand in range(num_hands):
        covered = float(0.0)  # noqa: UP018
        for basis in range(num_basis):
            best = float(0.0)  # noqa: UP018
            for body in range(num_bodies):
                index = ((world * num_hands + hand) * num_bodies + body) * num_basis + basis
                best = wp.max(best, current_support[index])
            if best > min_support:
                covered += 1.0
        # Our addition, arm CR: the simulator's own report that this hand is loading the object,
        # ungated by the reference label, so contact_consistency can disagree with the label.
        touching = float(0.0)  # noqa: UP018
        if covered > 0.0:
            touching = 1.0
        hand_touching[world * num_hands + hand] = touching
        if contact_active[frame * num_hands + hand] > 0.5:
            active_hands += 1.0
            reward += covered / float(num_basis)
""",
        where,
    )

    # (7) the composer's new inputs. Inserted immediately after anchors that no other patch
    # removes, and in the same relative order in the signature and at the launch, so applying
    # this arm and the orientation arm in either order still lines them up.
    text = replace(
        text,
        "    palm_orientation_std: float,\n",
        "    palm_orientation_std: float,\n"
        "    contact_role: wp.array(dtype=wp.float32),\n"
        "    hand_touching: wp.array(dtype=wp.float32),\n"
        "    object_radius: float,\n"
        "    contact_role_palm_std: float,\n"
        "    body_joint_position_in_contact_scale: float,\n",
        f"{where} (kernel signature)",
    )

    # (8) the terms themselves, and the frame-dependent body weight. Anchored on the start of the
    # weighted sum, which is after every term has been written, whatever the term count is.
    text = replace(
        text,
        "    total = float(0.0)  # noqa: UP018\n    for term in range(_RECON_BODY_OBJECTIVE_TERM_COUNT):",
        f"""    # Our addition, arm CR. Both palms are rewarded by palm_position on every frame, so an
    # idle hand is tracked as hard as the hand carrying the object. These two terms are gated on
    # the per-hand role: a contact-role hand is rewarded for closing the gap to the object's
    # surface, and again for the simulator agreeing that it is loading it; a free hand is not
    # rewarded at all. Averaged over the contact-role hands, so both terms stay in [0, 1].
    contact_role_reward = float(0.0)  # noqa: UP018
    contact_role_touching = float(0.0)  # noqa: UP018
    contact_role_hands = float(0.0)  # noqa: UP018
    contact_object_center = wp.transform_get_translation(body_q[body_base + object_body_id])
    for contact_hand in range(2):
        if contact_role[frame * 2 + contact_hand] > 0.5:
            contact_role_hands += 1.0
            contact_palm_xf = body_q[body_base + wrist_body_ids[contact_hand]]
            contact_palm = wp.transform_point(contact_palm_xf, wrist_local_pos[contact_hand])
            contact_offset = contact_palm - contact_object_center
            contact_gap = wp.max(wp.length(contact_offset) - object_radius, 0.0)
            contact_role_reward += wp.exp(
                -(contact_gap * contact_gap) / (contact_role_palm_std * contact_role_palm_std)
            )
            contact_role_touching += hand_touching[world * 2 + contact_hand]
    if contact_role_hands > 0.0:
        terms[term_base + {first}] = contact_role_reward / contact_role_hands
        terms[term_base + {second}] = contact_role_touching / contact_role_hands
        # While a hand is supposed to be holding something, whole-body joint tracking is worth
        # less: the scale is in_contact_weight / stock_weight, so the effective weight on term 6
        # is exactly the asked-for one. set_weights is per curriculum stage, not per frame, so
        # scaling the term is the only way to say this.
        terms[term_base + 6] = terms[term_base + 6] * body_joint_position_in_contact_scale
    else:
        terms[term_base + {first}] = 0.0
        terms[term_base + {second}] = 0.0

    total = float(0.0)  # noqa: UP018
    for term in range(_RECON_BODY_OBJECTIVE_TERM_COUNT):""",
        f"{where} (terms)",
    )

    # (9) the objective dataclass carries the role labels, the per-hand touch report, and the
    # object radius it reads once at build time.
    text = replace(
        text,
        "    contact_active: wp.array\n",
        "    contact_active: wp.array\n    contact_role: wp.array\n"
        "    hand_touching: wp.array\n    object_radius_m: float\n",
        f"{where} (dataclass fields)",
    )

    # (10) build(): the roles, and the radius. CommandBuffers.object_radius is what NVIDIA's own
    # wrench-support launch uses for the object's extent; it is constant for a rigid object, so it
    # is read once here rather than synced to the host every step.
    text = replace(
        text,
        "        basis = np.concatenate(\n",
        """        roles = _v2d_contact_roles(labels, reference.num_frames, len(config.sides))
        object_radius = getattr(command, "object_radius", None)
        if object_radius is None:
            raise ValueError(
                "ReconBody contact-role arm needs CommandBuffers.object_radius, which this "
                "flash_chord does not have; re-derive the arm against the installed source"
            )
        radius = np.asarray(getattr(object_radius, "numpy", lambda: object_radius)()).reshape(-1)
        if radius.size < 1 or not np.all(np.isfinite(radius)) or np.any(radius < 0.0):
            raise ValueError(f"ReconBody contact-role arm needs a finite object radius, got {radius!r}")
        basis = np.concatenate(
""",
        f"{where} (build roles)",
    )
    text = replace(
        text,
        "            contact_active=wp.array(labels.reshape(-1), dtype=wp.float32, device=device),\n",
        "            contact_active=wp.array(labels.reshape(-1), dtype=wp.float32, device=device),\n"
        "            contact_role=wp.array(roles.reshape(-1), dtype=wp.float32, device=device),\n"
        "            hand_touching=wp.zeros(\n"
        "                world_count * len(config.sides), dtype=wp.float32, device=device\n"
        "            ),\n"
        "            object_radius_m=float(radius[0]),\n",
        f"{where} (build fields)",
    )

    # (11) the launches. The reducer gains its second output; the composer gains the five inputs
    # from (7), in that order.
    text = replace(
        text,
        "            outputs=[self.force_closure],\n",
        "            outputs=[self.force_closure, self.hand_touching],\n",
        f"{where} (reduce launch)",
    )
    text = replace(
        text,
        "                self.config.palm_orientation_std,\n",
        "                self.config.palm_orientation_std,\n"
        "                self.contact_role,\n"
        "                self.hand_touching,\n"
        "                self.object_radius_m,\n"
        "                self.config.contact_role_palm_std,\n"
        "                self.config.body_joint_position_weight_in_contact\n"
        "                / max(self.config.body_joint_position_weight, 1e-12),\n",
        f"{where} (compose launch)",
    )
    return text


def objective_yaml_edits(
    weight: float, std: float, consistency_weight: float, body_weight_in_contact: float
) -> tuple[tuple[str, str], ...]:
    return (
        (
            "force_closure_weight: 5.0\n",
            "force_closure_weight: 5.0\n"
            f"contact_role_palm_weight: {num(weight)}\n"
            f"contact_consistency_weight: {num(consistency_weight)}\n",
        ),
        (
            "force_closure_min_support: 0.01\n",
            f"contact_role_palm_std: {num(std)}\n"
            f"body_joint_position_weight_in_contact: {num(body_weight_in_contact)}\n"
            "force_closure_min_support: 0.01\n",
        ),
    )


# set_weights() compares key SETS, so every curriculum stage that names the weights must name the
# new terms too, or the run dies at the first stage change. The ReconBody weight block is selected
# by `body_joint_position:` -- not by `force_closure:`, which three other objectives also use, and
# whose foreign files made an earlier patch refuse and cost seven clip claims.
CURRICULUM_ANCHOR = "      force_closure: 5.0\n"
CURRICULUM_SELECTOR = "      body_joint_position:"
def curriculum_edit(weight: float, consistency_weight: float) -> tuple[str, str]:
    return (
        CURRICULUM_ANCHOR,
        CURRICULUM_ANCHOR
        + f"      contact_role_palm: {num(weight)}\n"
        + f"      contact_consistency: {num(consistency_weight)}\n",
    )


def carried(path: Path) -> dict[str, float]:
    """The arm's settings a tree already has, read back from the objective yaml."""
    if not path.is_file():
        return {}
    text = path.read_text()
    out: dict[str, float] = {}
    for key in YAML_KEYS:
        found = re.search(rf"^{key}:\s*([0-9.eE+-]+)\s*$", text, re.MULTILINE)
        if found:
            out[key] = float(found.group(1))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root", type=Path, help="flash_chord source root, e.g. /workspace/src/flash_chord"
    )
    parser.add_argument(
        "--check", action="store_true", help="report what would change, write nothing"
    )
    parser.add_argument(
        "--weight", type=float, default=DEFAULT_WEIGHT,
        help=f"contact_role_palm weight (default {DEFAULT_WEIGHT:g})",
    )
    parser.add_argument(
        "--std", type=float, default=DEFAULT_STD,
        help=f"palm-to-surface shaping std in metres (default {DEFAULT_STD:g})",
    )
    parser.add_argument(
        "--consistency-weight", type=float, default=DEFAULT_CONSISTENCY_WEIGHT,
        help=f"contact_consistency weight (default {DEFAULT_CONSISTENCY_WEIGHT:g})",
    )
    parser.add_argument(
        "--body-weight-in-contact", type=float, default=DEFAULT_BODY_WEIGHT_IN_CONTACT,
        help=(
            "effective body_joint_position weight on contact-role frames "
            f"(default {DEFAULT_BODY_WEIGHT_IN_CONTACT:g}; stock is 5.0)"
        ),
    )
    args = parser.parse_args(argv)

    asked = {
        "contact_role_palm_weight": args.weight,
        "contact_role_palm_std": args.std,
        "contact_consistency_weight": args.consistency_weight,
        "body_joint_position_weight_in_contact": args.body_weight_in_contact,
    }
    for key, value in asked.items():
        if value != value or value in (float("inf"), float("-inf")):
            raise Refused(f"FATAL: {YAML_KEYS[key]} must be finite, got {value!r}")
    if not args.std > 0.0:
        raise Refused(f"FATAL: --std must be positive, got {args.std!r}")
    if args.body_weight_in_contact < 0.0:
        raise Refused(
            "FATAL: --body-weight-in-contact must be non-negative, got "
            f"{args.body_weight_in_contact!r}"
        )

    objective = args.root / "objectives/recon_body.py"
    objective_yaml = args.root / "configs/objective/recon_body.yaml"
    for path in (objective, objective_yaml):
        if not path.is_file():
            raise Refused(f"FATAL: {path} does not exist")

    have = carried(objective_yaml)
    mismatched = {k: v for k, v in have.items() if abs(v - asked[k]) > 1e-12}
    if mismatched:
        lines = "\n".join(
            f"    {k}: tree has {v:g}, asked for {asked[k]:g} ({YAML_KEYS[k]})"
            for k, v in sorted(mismatched.items())
        )
        raise Refused(
            f"FATAL: {objective_yaml} already carries different contact-role settings:\n{lines}\n"
            "  The patch is idempotent, so it would leave the first values in place and report\n"
            "  success, and the arm on the record would not be the arm that trained. Rebuild the\n"
            "  container, or ask for the values it already has."
        )

    curriculum_dir = args.root / "configs/curriculum"
    edit = curriculum_edit(args.weight, args.consistency_weight)
    curricula, foreign = [], []
    for path in sorted(curriculum_dir.glob("*.yaml")):
        text = path.read_text()
        if CURRICULUM_SELECTOR in text:
            curricula.append(path)
        elif "force_closure:" in text:
            foreign.append(path)
    if not curricula:
        raise Refused(
            "FATAL: no curriculum config carries the ReconBody weight block "
            f"({CURRICULUM_SELECTOR.strip()!r}); patch is incomplete"
        )

    # Plan every file before writing any of them: a tree with a 14-term objective and a 12-key
    # curriculum trains for hours and then dies at the first stage change, having billed for it.
    planned: list[tuple[Path, str]] = []
    report: list[str] = []
    if MARKER in objective.read_text():
        report.append(f"already patched: {objective}")
    else:
        patched = objective_edits(
            objective.read_text(),
            args.weight,
            args.std,
            args.consistency_weight,
            args.body_weight_in_contact,
            str(objective),
        )
        ast.parse(patched)  # a syntax error here would surface as a container crash minutes later
        planned.append((objective, patched))
        report.append(f"patched: {objective}")
    for path, edits in [(objective_yaml, objective_yaml_edits(
        args.weight, args.std, args.consistency_weight, args.body_weight_in_contact
    ))]:
        text = path.read_text()
        if MARKER in text:
            report.append(f"already patched: {path}")
            continue
        for old, new in edits:
            text = replace(text, old, new, str(path))
        planned.append((path, text))
        report.append(f"patched: {path}")
    for path in curricula:
        text = path.read_text()
        if MARKER in text:
            report.append(f"already patched: {path}")
            continue
        text = replace(text, edit[0], edit[1], str(path))
        planned.append((path, text))
        report.append(f"patched: {path}")

    if args.check:
        report = [message.replace("patched:", "would patch:") for message in report]
    else:
        for path, text in planned:
            path.write_text(text)

    print("\n".join(report))
    for path in foreign:
        print(
            f"left alone: {path} -- names force_closure but not the ReconBody weight block, so it "
            "is a different objective"
        )
    print(
        f"contact-role arm: contact_role_palm weight {args.weight:g} std {args.std:g} m, "
        f"contact_consistency weight {args.consistency_weight:g}, body_joint_position "
        f"{args.body_weight_in_contact:g} while in contact"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
