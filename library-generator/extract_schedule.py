#!/usr/bin/env python3
import argparse
import json
import re
import sys


ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


def _strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def _find_value(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    return None


def _parse_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"Expected boolean, got: {value}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract FuseFlow schedule flags from Honeybee output."
    )
    parser.add_argument(
        "--format",
        choices=["flags", "json", "kv"],
        default="flags",
        help="Output format (default: flags)",
    )
    args = parser.parse_args()

    text = _strip_ansi(sys.stdin.read())

    fields = {
        "mlir": _find_value(r'mlir\s*=\s*"([^"]+)"', text),
        "stream_level": _find_value(r"stream_level\s*=\s*(\d+)", text),
        "par_factor": _find_value(r"par_factor\s*=\s*(\d+)", text),
        "stream_shape": _find_value(r"stream_shape\s*=\s*(\d+)", text),
        "block_sparse": _find_value(r"block_sparse\s*=\s*(True|False)", text),
    }

    missing = [key for key, value in fields.items() if value is None]
    if missing:
        sys.stderr.write(
            "Missing schedule fields in Honeybee output: "
            + ", ".join(missing)
            + "\n"
        )
        return 1

    schedule = {
        "mlir": fields["mlir"],
        "stream_level": int(fields["stream_level"]),
        "par_factor": int(fields["par_factor"]),
        "stream_shape": int(fields["stream_shape"]),
        "block_sparse": _parse_bool(fields["block_sparse"]),
    }

    if args.format == "json":
        print(json.dumps(schedule))
        return 0

    if args.format == "kv":
        for key, value in schedule.items():
            print(f"{key}={value}")
        return 0

    par_flag = (
        "--stream-parallelizer="
        f"stream-level={schedule['stream_level']} "
        f"par-factor={schedule['par_factor']}"
    )
    vec_flag = f"--stream-vectorizer=stream-shape={schedule['stream_shape']}"
    if schedule["block_sparse"]:
        vec_flag += " enable-block-sparse"

    print(par_flag)
    print(vec_flag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
