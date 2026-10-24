"""Rewrite the absolute asset paths a retargeted motion parquet embeds.

`soma_to_g1.py` records `object_mesh_paths` and `object_urdf_paths` as absolute paths on the
machine that produced them. Moving the parquet to another machine therefore breaks it, and it
breaks in a way that names neither the path nor the machine: Newton's `add_urdf(source=...)`
cannot stat the missing file, falls back to treating the string as inline URDF XML, and raises

    xml.etree.ElementTree.ParseError: not well-formed (invalid token): line 1, column 0

which is the path being parsed as a document. That cost a dev-box iteration to find.

Rebasing is a copy, never in place: the original stays the record of what was retargeted.
"""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath

import pyarrow as pa
import pyarrow.parquet as pq

ASSET_COLUMNS = ("object_mesh_paths", "object_urdf_paths")


def rebase(table: pa.Table, new_root: str) -> tuple[pa.Table, dict[str, list[str]]]:
    """Point every asset path at `new_root`, keeping only each path's file name.

    The retargeter writes all of an episode's object assets into one directory, so the file name
    is the only part worth keeping; `new_root` is where that directory lives on the target.
    """
    changed: dict[str, list[str]] = {}
    for column in ASSET_COLUMNS:
        if column not in table.column_names:
            continue
        original = table[column].to_pylist()
        rewritten = []
        for row in original:
            if row is None:
                rewritten.append(row)
                continue
            moved = [str(PurePosixPath(new_root) / PurePosixPath(str(path)).name) for path in row]
            rewritten.append(moved)
        changed[column] = [p for row in rewritten if row for p in row]
        index = table.column_names.index(column)
        table = table.set_column(
            index, column, pa.array(rewritten, type=table.schema.field(column).type)
        )
    return table, changed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="v2d rebase-assets",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("parquet", type=Path, help="a retargeted motion_v1 parquet")
    parser.add_argument(
        "--new-root",
        required=True,
        help="directory holding the object assets on the TARGET machine",
    )
    parser.add_argument("-o", "--out", type=Path, required=True, help="where to write the copy")
    args = parser.parse_args(argv)

    table = pq.read_table(args.parquet)
    rebased, changed = rebase(table, args.new_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(rebased, args.out)
    for column, paths in changed.items():
        for path in paths:
            print(f"{column}: {path}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
