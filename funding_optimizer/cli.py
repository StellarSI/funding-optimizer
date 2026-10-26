"""Command line interface for funding-optimizer.

The ``optimize`` command reads a budget and programs from JSON, solves the
allocation problem, and writes the plan as JSON. It only recommends
allocations; it never signs or submits payouts.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import cast

from funding_optimizer import __version__
from funding_optimizer.errors import ValidationError
from funding_optimizer.models import Budget, Program
from funding_optimizer.optimizer import FAIRNESS_MODES, OBJECTIVES, Optimizer
from funding_optimizer.plan import AllocationPlan

__all__ = ["build_parser", "main"]

_PROGRAM_FIELDS = frozenset({"id", "min_share", "expected_reach", "region"})


def _expect_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValidationError(f"{label} must be a JSON object")
    return cast(Mapping[str, object], value)


def _expect_list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValidationError(f"{label} must be a JSON array")
    return cast(list[object], value)


def _read_json(path: str) -> object:
    if path == "-":
        text = sys.stdin.read()
    else:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"invalid JSON in {path!r}: {exc.msg}") from exc


def _split_payload(
    payload: object,
) -> tuple[object | None, list[object], str | None, str | None]:
    if isinstance(payload, list):
        return None, cast(list[object], payload), None, None
    if not isinstance(payload, dict):
        raise ValidationError("input JSON must be an object or an array of programs")

    data = cast(Mapping[str, object], payload)
    if "programs" not in data:
        raise ValidationError("input JSON must contain a 'programs' array")
    programs = _expect_list(data["programs"], "programs")

    objective = data.get("objective")
    fairness = data.get("fairness")
    if objective is not None and not isinstance(objective, str):
        raise ValidationError("objective must be a string")
    if fairness is not None and not isinstance(fairness, str):
        raise ValidationError("fairness must be a string")

    return data.get("budget"), programs, objective, fairness


def _budget_from_payload(
    payload: object | None, total_override: float | None, asset_override: str | None
) -> Budget:
    total: object | None = total_override
    asset: object | None = asset_override

    if payload is not None:
        data = _expect_mapping(payload, "budget")
        if total is None:
            total = data.get("total")
        if asset is None:
            asset = data.get("asset")

    if total is None or asset is None:
        raise ValidationError("budget requires 'total' and 'asset' (or pass --total and --asset)")
    return Budget(total=cast(float, total), asset=cast(str, asset))


def _program_from_payload(payload: object) -> Program:
    data = _expect_mapping(payload, "each program")
    unknown = sorted(str(field) for field in set(data) - _PROGRAM_FIELDS)
    if unknown:
        raise ValidationError(f"unknown program field(s): {', '.join(unknown)}")
    return Program(
        id=cast(str, data.get("id")),
        min_share=cast(float, data.get("min_share", 0.0)),
        expected_reach=cast(int, data.get("expected_reach", 0)),
        region=cast(str | None, data.get("region")),
    )


def _solve_from_payload(payload: object, args: argparse.Namespace) -> AllocationPlan:
    budget_payload, programs_payload, payload_objective, payload_fairness = _split_payload(payload)
    budget = _budget_from_payload(budget_payload, args.total, args.asset)
    programs = [_program_from_payload(item) for item in programs_payload]

    objective = args.objective if args.objective is not None else payload_objective
    fairness = args.fairness if args.fairness is not None else payload_fairness
    return Optimizer(
        objective="maximize_reach" if objective is None else objective,
        fairness="none" if fairness is None else fairness,
    ).solve(budget, programs)


def _write_plan(plan: AllocationPlan, output: str, indent: int | None) -> None:
    text = json.dumps(asdict(plan), indent=indent, ensure_ascii=False) + "\n"
    if output == "-":
        sys.stdout.write(text)
        return
    with open(output, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="funding-optimizer",
        description="Recommend grant and disbursement allocations.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    optimize = subparsers.add_parser(
        "optimize",
        help="solve a JSON allocation problem and write the plan",
        description=(
            "Read a budget and programs from JSON, solve the allocation problem, "
            "and write the plan as JSON."
        ),
    )
    optimize.add_argument(
        "--input",
        "-i",
        required=True,
        metavar="PATH",
        help="program JSON file, or '-' for stdin",
    )
    optimize.add_argument(
        "--output",
        "-o",
        default="-",
        metavar="PATH",
        help="plan JSON file, or '-' for stdout (default: -)",
    )
    optimize.add_argument(
        "--total",
        "--budget-total",
        type=float,
        default=None,
        metavar="AMOUNT",
        help="budget total; overrides the value in the input file",
    )
    optimize.add_argument(
        "--asset",
        default=None,
        metavar="CODE",
        help="budget asset; overrides the value in the input file",
    )
    optimize.add_argument(
        "--objective",
        choices=sorted(OBJECTIVES),
        default=None,
        help="override the objective from the input file",
    )
    optimize.add_argument(
        "--fairness",
        choices=sorted(FAIRNESS_MODES),
        default=None,
        help="override the fairness mode from the input file",
    )
    optimize.add_argument(
        "--indent",
        type=int,
        default=2,
        metavar="N",
        help="JSON indentation width (default: 2)",
    )
    optimize.add_argument(
        "--compact",
        action="store_true",
        help="write compact JSON instead of indented JSON",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = _read_json(args.input)
        plan = _solve_from_payload(payload, args)
        _write_plan(plan, args.output, None if args.compact else args.indent)
    except (ValidationError, OSError, UnicodeDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
