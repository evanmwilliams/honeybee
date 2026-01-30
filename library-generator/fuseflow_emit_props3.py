#!/usr/bin/env python3
import argparse
import os
import re
import sys


ITERATOR_RE = re.compile(r"iterator_types\s*=\s*\[([^\]]+)\]")


def _parse_iterator_types(text: str) -> list[list[str]]:
    lists: list[list[str]] = []
    for match in ITERATOR_RE.finditer(text):
        items = []
        raw = match.group(1)
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            if part.startswith('"') and part.endswith('"'):
                part = part[1:-1]
            items.append(part)
        if items:
            lists.append(items)
    return lists


def _infer_loop_info(text: str) -> int:
    iterator_lists = _parse_iterator_types(text)
    max_loops = 0
    for items in iterator_lists:
        max_loops = max(max_loops, len(items))
    if max_loops == 0 and "linalg.matmul" in text:
        max_loops = 3
    return max_loops


def _emit_prop(name: str, args: dict[str, object]) -> None:
    print("[[Prop]]")
    print(f'name = "{name}"')
    for key, val in args.items():
        if isinstance(val, str):
            print(f'args.{key} = "{val}"')
        elif isinstance(val, bool):
            print(f'args.{key} = {str(val).lower()}')
        else:
            print(f'args.{key} = {val}')
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Emit Honeybee Prop facts for FuseFlow MLIR inputs."
    )
    parser.add_argument("mlir", help="Path to input MLIR file")
    parser.add_argument(
        "--mlir-id",
        help="Stable MLIR identifier (default: file stem)",
    )
    parser.add_argument(
        "--no-mlir-input",
        action="store_true",
        help="Do not emit the MlirInput prop",
    )
    args = parser.parse_args()

    with open(args.mlir, "r", encoding="utf-8") as f:
        text = f.read()

    mlir_path = os.path.abspath(args.mlir)
    mlir_id = args.mlir_id or os.path.splitext(os.path.basename(args.mlir))[0]

    max_loops = _infer_loop_info(text)

    if not args.no_mlir_input:
        _emit_prop(
            "MlirInput",
            {
                "id": mlir_id,
                "path": mlir_path,
                "num_loops": max_loops,
            },
        )

    print("[Goal]")
    print('name = "ScheduleChoice"')
    print(f'args.mlir = "{mlir_id}"')
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
