"""CSV loader for programs, cohorts, and allocation constraints (issue #31).

Reads operator-maintained CSV exports — the program list, cohort membership,
and regional fairness floors — into the typed structures the optimizer
consumes. Parsing is strict and deterministic:

- Files are read as UTF-8 (a BOM is tolerated); badly encoded bytes raise.
- The header row is required and matched case-insensitively and
  whitespace-trimmed; missing required columns are reported by name.
  Unknown columns are ignored so exports can carry extra columns.
- Every row is validated; failures raise :class:`CsvLoaderError` naming the
  row number and the offending field. Empty lines are skipped.
- Numbers must be finite; shares (``min_share``, ``floor``, cohort bounds)
  must lie in ``[0, 1]``; identifiers and regions must be non-empty after
  trimming.
- Duplicate program or cohort ids are rejected; cohort programs must exist
  in the loaded program list (when one is supplied), and a program may
  belong to at most one cohort; cohort ``min_share`` may not exceed
  ``max_share``, and cohort ``min_share`` values may not sum over 1.0.

The optimizer recommends allocations and never signs or submits payouts.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import TextIO

from funding_optimizer.optimizer.cohorts import Cohort
from funding_optimizer.optimizer.constraints import Program, RegionFloor


class CsvLoaderError(ValueError):
    """Raised when a CSV input file cannot be parsed or validated."""


_PROGRAM_COLUMNS = ("id", "region", "expected_reach")
_COHORT_COLUMNS = ("id", "program_ids")
_FLOOR_COLUMNS = ("region", "floor")


@dataclass(frozen=True)
class AllocationInputs:
    """Typed bundle of everything a single allocation run needs."""

    programs: tuple[Program, ...]
    cohorts: tuple[Cohort, ...]
    region_floors: tuple[RegionFloor, ...]


def _read_rows(
    source: str | Path | TextIO,
    *,
    required_columns: Sequence[str],
    label: str,
) -> tuple[list[dict[str, str]], list[int]]:
    """Parse ``source`` into rows with a validated header.

    Returns the rows (with normalized header keys) and the 1-based line
    numbers they came from so validation errors can point at the file.
    """
    close = False
    if isinstance(source, (str, Path)):
        try:
            handle: TextIO = open(source, newline="", encoding="utf-8-sig")
        except OSError as exc:
            raise CsvLoaderError(f"{label}: cannot open {source!s}: {exc}") from exc
        close = True
    else:
        handle = source

    try:
        try:
            raw_rows = list(csv.reader(handle))
        except UnicodeDecodeError as exc:
            raise CsvLoaderError(f"{label}: file is not valid UTF-8: {exc}") from exc
        except csv.Error as exc:
            raise CsvLoaderError(f"{label}: malformed CSV: {exc}") from exc
    finally:
        if close:
            handle.close()

    if not raw_rows:
        raise CsvLoaderError(f"{label}: file is empty; a header row is required")

    header = [cell.strip().lower() for cell in raw_rows[0]]
    if any(cell == "" for cell in header):
        raise CsvLoaderError(f"{label}: header contains an empty column name")
    normalized: list[str] = []
    for cell in header:
        if cell in normalized:
            raise CsvLoaderError(f"{label}: header contains duplicate column {cell!r}")
        normalized.append(cell)

    missing = [column for column in required_columns if column not in normalized]
    if missing:
        raise CsvLoaderError(
            f"{label}: missing required column(s) {', '.join(missing)}; "
            f"found: {', '.join(normalized) or '(none)'}"
        )

    rows: list[dict[str, str]] = []
    line_numbers: list[int] = []
    for offset, raw in enumerate(raw_rows[1:], start=2):
        if not raw or all(cell.strip() == "" for cell in raw):
            continue
        record = dict(zip(normalized, (cell.strip() for cell in raw)))
        if len(raw) > len(normalized):
            # Record extra cells so nothing silently disappears.
            extras = [cell.strip() for cell in raw[len(normalized):]]
            if any(extras):
                raise CsvLoaderError(
                    f"{label} row {offset}: expected at most {len(normalized)} "
                    f"column(s), got {len(raw)}"
                )
        elif len(raw) < len(normalized):
            record.update({column: "" for column in normalized[len(raw):]})
        rows.append(record)
        line_numbers.append(offset)
    return rows, line_numbers


def _parse_float(value: str, *, field: str, label: str, row: int) -> float:
    if value == "":
        raise CsvLoaderError(f"{label} row {row}: {field} is required")
    try:
        parsed = float(value)
    except ValueError:
        raise CsvLoaderError(
            f"{label} row {row}: {field} must be a number, got {value!r}"
        ) from None
    if not isfinite(parsed):
        raise CsvLoaderError(f"{label} row {row}: {field} must be finite, got {value!r}")
    return parsed


def _parse_share(value: str, *, field: str, label: str, row: int) -> float:
    parsed = _parse_float(value, field=field, label=label, row=row)
    if not 0.0 <= parsed <= 1.0:
        raise CsvLoaderError(f"{label} row {row}: {field} must be in [0, 1], got {parsed}")
    return parsed


def _require_text(value: str, *, field: str, label: str, row: int) -> str:
    if value == "":
        raise CsvLoaderError(f"{label} row {row}: {field} must not be empty")
    return value


def load_programs(source: str | Path | TextIO) -> list[Program]:
    """Load programs from a CSV export.

    Expected columns: ``id``, ``region``, ``expected_reach``, and the
    optional ``min_share`` (a fraction in ``[0, 1]``; defaults to ``0``).
    """
    rows, line_numbers = _read_rows(
        source, required_columns=_PROGRAM_COLUMNS, label="programs"
    )
    programs: list[Program] = []
    seen: set[str] = set()
    for record, row in zip(rows, line_numbers):
        program_id = _require_text(record["id"], field="id", label="programs", row=row)
        if program_id in seen:
            raise CsvLoaderError(f"programs row {row}: duplicate program id {program_id!r}")
        seen.add(program_id)
        region = _require_text(record["region"], field="region", label="programs", row=row)
        expected_reach = _parse_float(
            record["expected_reach"], field="expected_reach", label="programs", row=row
        )
        if expected_reach < 0.0:
            raise CsvLoaderError(
                f"programs row {row}: expected_reach must be non-negative, got {expected_reach}"
            )
        min_share = _parse_share(
            record.get("min_share", "").strip() or "0",
            field="min_share",
            label="programs",
            row=row,
        )
        programs.append(
            Program(id=program_id, region=region, expected_reach=expected_reach,
                    min_share=min_share)
        )
    if not programs:
        raise CsvLoaderError("programs: no program rows found in input")
    return programs


def load_cohorts(
    source: str | Path | TextIO,
    *,
    programs: Iterable[Program] | None = None,
) -> list[Cohort]:
    """Load cohorts from a CSV export.

    Expected columns: ``id``, ``program_ids`` (member ids separated by
    ``;``), and the optional ``min_share`` / ``max_share`` (fractions in
    ``[0, 1]``; defaults ``0`` and ``1``).

    When ``programs`` is supplied, every referenced program must exist and
    a program may belong to at most one cohort. Cohort ``min_share``
    values may not sum over ``1.0``.
    """
    rows, line_numbers = _read_rows(
        source, required_columns=_COHORT_COLUMNS, label="cohorts"
    )
    known_ids: set[str] | None = None
    if programs is not None:
        known_ids = {program.id for program in programs}
        if not known_ids:
            raise CsvLoaderError("cohorts: programs must not be empty when supplied")

    cohorts: list[Cohort] = []
    seen_cohort_ids: set[str] = set()
    assigned: set[str] = set()
    total_min_share = 0.0
    for record, row in zip(rows, line_numbers):
        cohort_id = _require_text(record["id"], field="id", label="cohorts", row=row)
        if cohort_id in seen_cohort_ids:
            raise CsvLoaderError(f"cohorts row {row}: duplicate cohort id {cohort_id!r}")
        seen_cohort_ids.add(cohort_id)

        raw_members = _require_text(
            record["program_ids"], field="program_ids", label="cohorts", row=row
        )
        members = tuple(member.strip() for member in raw_members.split(";") if member.strip())
        if not members:
            raise CsvLoaderError(
                f"cohorts row {row}: program_ids must list at least one program id"
            )
        if len(set(members)) != len(members):
            raise CsvLoaderError(
                f"cohorts row {row}: program_ids contains duplicate entries"
            )
        for member in members:
            if known_ids is not None and member not in known_ids:
                raise CsvLoaderError(
                    f"cohorts row {row}: cohort {cohort_id!r} references unknown "
                    f"program {member!r}"
                )
            if member in assigned:
                raise CsvLoaderError(
                    f"cohorts row {row}: program {member!r} is assigned to multiple cohorts"
                )
            assigned.add(member)

        min_share = _parse_share(
            record.get("min_share", "").strip() or "0",
            field="min_share",
            label="cohorts",
            row=row,
        )
        max_share = _parse_share(
            record.get("max_share", "").strip() or "1",
            field="max_share",
            label="cohorts",
            row=row,
        )
        if min_share > max_share:
            raise CsvLoaderError(
                f"cohorts row {row}: cohort {cohort_id!r} min_share {min_share} "
                f"exceeds max_share {max_share}"
            )
        total_min_share += min_share
        if total_min_share > 1.0:
            raise CsvLoaderError(
                f"cohorts row {row}: cohort min_shares sum to {total_min_share}, "
                "which exceeds the whole budget (1.0)"
            )
        cohorts.append(
            Cohort(id=cohort_id, program_ids=members, min_share=min_share,
                   max_share=max_share)
        )
    return cohorts


def load_region_floors(source: str | Path | TextIO) -> list[RegionFloor]:
    """Load regional fairness floors from a constraints CSV export.

    Expected columns: ``region`` and ``floor`` (a fraction in ``[0, 1]``).
    Rows may repeat a region only when the floors agree; duplicate rows
    with different floors are rejected. Floor values may not sum over
    ``1.0``.
    """
    rows, line_numbers = _read_rows(
        source, required_columns=_FLOOR_COLUMNS, label="constraints"
    )
    floors: dict[str, float] = {}
    total = 0.0
    for record, row in zip(rows, line_numbers):
        region = _require_text(record["region"], field="region", label="constraints", row=row)
        floor = _parse_share(record["floor"], field="floor", label="constraints", row=row)
        if region in floors:
            if floors[region] != floor:
                raise CsvLoaderError(
                    f"constraints row {row}: region {region!r} has conflicting "
                    f"floors {floors[region]} and {floor}"
                )
            continue
        floors[region] = floor
        total += floor
        if total > 1.0:
            raise CsvLoaderError(
                f"constraints row {row}: region floors sum to {total}, "
                "which exceeds the whole budget (1.0)"
            )
    return [RegionFloor(region=region, floor=floor) for region, floor in floors.items()]


def load_allocation_inputs(
    programs: str | Path | TextIO,
    *,
    cohorts: str | Path | TextIO | None = None,
    constraints: str | Path | TextIO | None = None,
) -> AllocationInputs:
    """Load the complete input bundle for one allocation run.

    ``programs`` is required; ``cohorts`` and ``constraints`` are optional.
    Cohort membership is validated against the loaded programs, so unknown
    or doubly-assigned program ids are rejected before the optimizer sees
    them.
    """
    program_list = load_programs(programs)
    cohort_list = (
        load_cohorts(cohorts, programs=program_list) if cohorts is not None else []
    )
    floors = load_region_floors(constraints) if constraints is not None else []
    return AllocationInputs(
        programs=tuple(program_list),
        cohorts=tuple(cohort_list),
        region_floors=tuple(floors),
    )
