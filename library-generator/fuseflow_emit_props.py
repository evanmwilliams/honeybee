#!/usr/bin/env python3
import argparse
import os
import re
import sys


ITERATOR_RE = re.compile(r"iterator_types\s*=\s*\[([^\]]+)\]")


def _parse_int_list(csv: str) -> list[int]:
    out: list[int] = []
    for part in csv.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


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


def _infer_loop_info(text: str) -> tuple[int, set[int], set[int]]:
    iterator_lists = _parse_iterator_types(text)
    parallel_levels: set[int] = set()
    reduction_levels: set[int] = set()
    max_loops = 0
    for items in iterator_lists:
        max_loops = max(max_loops, len(items))
        for idx, it in enumerate(items):
            if it == "parallel":
                parallel_levels.add(idx)
            elif it == "reduction":
                reduction_levels.add(idx)
    if max_loops == 0 and "linalg.matmul" in text:
        max_loops = 3
        parallel_levels.update([0, 1])
        reduction_levels.add(2)
    return max_loops, parallel_levels, reduction_levels


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
        description="Emit Honeybee Prop facts for FuseFlow schedule options."
    )
    parser.add_argument("mlir", help="Path to input MLIR file")
    parser.add_argument(
        "--mlir-id",
        help="Stable MLIR identifier (default: file stem)",
    )
    parser.add_argument(
        "--par-factors",
        default="1,2,4,8",
        help="Comma-separated list of par-factors (default: 1,2,4,8)",
    )
    parser.add_argument(
        "--vec-shapes",
        default="1,2,4,8,16",
        help="Comma-separated list of stream shapes (default: 1,2,4,8,16)",
    )
    parser.add_argument(
        "--include-reduction-levels",
        action="store_true",
        help="Allow reduction loop positions for stream_level",
    )
    parser.add_argument(
        "--include-block-sparse",
        action="store_true",
        help="Emit block-sparse vectorization options",
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

    max_loops, parallel_levels, reduction_levels = _infer_loop_info(text)

    if not parallel_levels and max_loops > 0:
        parallel_levels = set(range(max_loops))

    if args.include_reduction_levels:
        parallel_levels.update(reduction_levels)

    par_factors = _parse_int_list(args.par_factors)
    vec_shapes = _parse_int_list(args.vec_shapes)

    if not args.no_mlir_input:
        _emit_prop(
            "MlirInput",
            {
                "id": mlir_id,
                "path": mlir_path,
                "num_loops": max_loops,
            },
        )

    for level in sorted(parallel_levels):
        for factor in par_factors:
            _emit_prop(
                "ParConfig",
                {
                    "mlir": mlir_id,
                    "level": level,
                    "factor": factor,
                },
            )

    block_sparse_vals = [False]
    if args.include_block_sparse:
        block_sparse_vals.append(True)

    for shape in vec_shapes:
        for block_sparse in block_sparse_vals:
            _emit_prop(
                "VecConfig",
                {
                    "mlir": mlir_id,
                    "shape": shape,
                    "block_sparse": block_sparse,
                },
            )

    print("[Goal]")
    print('name = "ScheduleChoice"')
    print(f'args.mlir = "{mlir_id}"')
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
