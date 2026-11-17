"""Tests for the CSV ingestion loader (issue #31).

Sanitized fixtures only; no live Horizon or analytics-api calls. All
parsers are strict: failures are reported with the offending row number
and field.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from funding_optimizer.ingest.csv_loader import (
    AllocationInputs,
    CsvLoaderError,
    load_allocation_inputs,
    load_cohorts,
    load_programs,
    load_region_floors,
)
from funding_optimizer.optimizer.constraints import Program

PROGRAMS_CSV = "id,region,expected_reach,min_share\n\
p1,Africa,12000,0.20\n\
p2,Asia,4500,\n\
p3,Africa,7800,0.10\n"


def test_load_programs_happy_path() -> None:
    programs = load_programs(io.StringIO(PROGRAMS_CSV))

    assert [program.id for program in programs] == ["p1", "p2", "p3"]
    assert programs[0].region == "Africa"
    assert programs[0].expected_reach == pytest.approx(12000.0)
    assert programs[0].min_share == pytest.approx(0.20)
    # Missing min_share defaults to 0.
    assert programs[1].min_share == pytest.approx(0.0)


def test_load_programs_trims_whitespace_and_accepts_bom(tmp_path: Path) -> None:
    path = tmp_path / "programs.csv"
    path.write_text(
        "\ufeff id , region , expected_reach , min_share \np9 , Europe , 3300 , 0.05\n",
        encoding="utf-8",
    )
    programs = load_programs(path)

    assert programs[0].id == "p9"
    assert programs[0].region == "Europe"
    assert programs[0].min_share == pytest.approx(0.05)


def test_load_programs_missing_column_reports_name() -> None:
    with pytest.raises(CsvLoaderError, match="missing required column"):
        load_programs(io.StringIO("id,region\np1,Africa\n"))


def test_load_programs_empty_file_is_rejected() -> None:
    with pytest.raises(CsvLoaderError, match="empty"):
        load_programs(io.StringIO(""))


def test_load_programs_malformed_number_names_row() -> None:
    csv_text = "id,region,expected_reach\np1,Africa,not-a-number\n"
    with pytest.raises(CsvLoaderError, match="row 2"):
        load_programs(io.StringIO(csv_text))


def test_load_programs_share_out_of_range_rejected() -> None:
    csv_text = "id,region,expected_reach,min_share\np1,Africa,100,1.5\n"
    with pytest.raises(CsvLoaderError, match="row 2"):
        load_programs(io.StringIO(csv_text))


def test_load_programs_negative_reach_rejected() -> None:
    csv_text = "id,region,expected_reach\np1,Africa,-1\n"
    with pytest.raises(CsvLoaderError, match="row 2"):
        load_programs(io.StringIO(csv_text))


def test_load_programs_duplicate_id_rejected() -> None:
    csv_text = "id,region,expected_reach\np1,Africa,100\np1,Asia,200\n"
    with pytest.raises(CsvLoaderError, match="duplicate program id"):
        load_programs(io.StringIO(csv_text))


def test_load_programs_empty_id_rejected() -> None:
    csv_text = "id,region,expected_reach\n,Africa,100\n"
    with pytest.raises(CsvLoaderError, match="row 2"):
        load_programs(io.StringIO(csv_text))


def test_load_programs_extra_columns_are_ignored() -> None:
    csv_text = "id,region,expected_reach,notes\np1,Africa,100,managed by NGO\n"
    programs = load_programs(io.StringIO(csv_text))
    assert programs[0].id == "p1"


def test_load_programs_too_many_cells_rejected() -> None:
    csv_text = "id,region,expected_reach\np1,Africa,100,x\n"
    with pytest.raises(CsvLoaderError, match="row 2"):
        load_programs(io.StringIO(csv_text))


def test_load_programs_missing_cells_become_empty_and_fail_clearly() -> None:
    csv_text = "id,region,expected_reach\np1\n"
    with pytest.raises(CsvLoaderError, match="row 2"):
        load_programs(io.StringIO(csv_text))


COHORTS_CSV = "id,program_ids,min_share,max_share\n\
c1,\"p1;p2\",0.1,0.8\n\
c2,p3,0.0,1.0\n"


def test_load_cohorts_happy_path() -> None:
    cohorts = load_cohorts(io.StringIO(COHORTS_CSV))

    assert [cohort.id for cohort in cohorts] == ["c1", "c2"]
    assert cohorts[0].program_ids == ("p1", "p2")
    assert cohorts[0].min_share == pytest.approx(0.1)
    assert cohorts[0].max_share == pytest.approx(0.8)


def test_load_cohorts_validates_membership_against_programs() -> None:
    programs = load_programs(io.StringIO(PROGRAMS_CSV))
    cohorts = load_cohorts(io.StringIO(COHORTS_CSV), programs=programs)
    assert cohorts[0].program_ids == ("p1", "p2")

    bad = "id,program_ids\nc1,\"p1;unknown\"\n"
    with pytest.raises(CsvLoaderError, match="unknown program"):
        load_cohorts(io.StringIO(bad), programs=programs)


def test_load_cohorts_rejects_double_assignment() -> None:
    csv_text = "id,program_ids\nc1,\"p1;p2\"\nc2,p2\n"
    with pytest.raises(CsvLoaderError, match="multiple cohorts"):
        load_cohorts(io.StringIO(csv_text))


def test_load_cohorts_min_share_above_max_share_rejected() -> None:
    csv_text = "id,program_ids,min_share,max_share\nc1,p1,0.9,0.2\n"
    with pytest.raises(CsvLoaderError, match="exceeds max_share"):
        load_cohorts(io.StringIO(csv_text))


def test_load_cohorts_min_shares_cannot_sum_over_one() -> None:
    csv_text = "id,program_ids,min_share\nc1,p1,0.7\nc2,p2,0.4\n"
    with pytest.raises(CsvLoaderError, match="sum"):
        load_cohorts(io.StringIO(csv_text))


def test_load_cohorts_empty_member_list_rejected() -> None:
    csv_text = "id,program_ids\nc1,\";;;\"\n"
    with pytest.raises(CsvLoaderError, match="at least one program"):
        load_cohorts(io.StringIO(csv_text))


def test_load_cohorts_duplicate_member_rejected() -> None:
    csv_text = "id,program_ids\nc1,\"p1;p1\"\n"
    with pytest.raises(CsvLoaderError, match="duplicate"):
        load_cohorts(io.StringIO(csv_text))


def test_load_cohorts_duplicate_id_rejected() -> None:
    csv_text = "id,program_ids\nc1,p1\nc1,p2\n"
    with pytest.raises(CsvLoaderError, match="duplicate cohort id"):
        load_cohorts(io.StringIO(csv_text))


FLOORS_CSV = "region,floor\nAfrica,0.5\nAsia,0.2\n"


def test_load_region_floors_happy_path() -> None:
    floors = load_region_floors(io.StringIO(FLOORS_CSV))

    assert [(f.region, f.floor) for f in floors] == [("Africa", 0.5), ("Asia", 0.2)]


def test_load_region_floors_conflicting_duplicates_rejected() -> None:
    csv_text = "region,floor\nAfrica,0.5\nAfrica,0.6\n"
    with pytest.raises(CsvLoaderError, match="conflicting"):
        load_region_floors(io.StringIO(csv_text))


def test_load_region_floors_identical_duplicates_collapse() -> None:
    csv_text = "region,floor\nAfrica,0.5\nAfrica,0.5\n"
    floors = load_region_floors(io.StringIO(csv_text))
    assert [(f.region, f.floor) for f in floors] == [("Africa", 0.5)]


def test_load_region_floors_sum_over_one_rejected() -> None:
    csv_text = "region,floor\nAfrica,0.7\nAsia,0.4\n"
    with pytest.raises(CsvLoaderError, match="sum"):
        load_region_floors(io.StringIO(csv_text))


def test_load_region_floors_missing_floor_value_rejected() -> None:
    csv_text = "region,floor\nAfrica,\n"
    with pytest.raises(CsvLoaderError, match="row 2"):
        load_region_floors(io.StringIO(csv_text))


def test_load_allocation_inputs_bundle(tmp_path: Path) -> None:
    programs_path = tmp_path / "programs.csv"
    cohorts_path = tmp_path / "cohorts.csv"
    constraints_path = tmp_path / "constraints.csv"
    programs_path.write_text(PROGRAMS_CSV, encoding="utf-8")
    cohorts_path.write_text(COHORTS_CSV, encoding="utf-8")
    constraints_path.write_text(FLOORS_CSV, encoding="utf-8")

    bundle = load_allocation_inputs(
        programs_path, cohorts=cohorts_path, constraints=constraints_path
    )

    assert isinstance(bundle, AllocationInputs)
    assert [p.id for p in bundle.programs] == ["p1", "p2", "p3"]
    assert [c.id for c in bundle.cohorts] == ["c1", "c2"]
    assert [f.region for f in bundle.region_floors] == ["Africa", "Asia"]


def test_load_allocation_inputs_rejects_bad_cohort_membership(tmp_path: Path) -> None:
    programs_path = tmp_path / "programs.csv"
    cohorts_path = tmp_path / "cohorts.csv"
    programs_path.write_text(PROGRAMS_CSV, encoding="utf-8")
    cohorts_path.write_text("id,program_ids\nc1,nope\n", encoding="utf-8")

    with pytest.raises(CsvLoaderError, match="unknown program"):
        load_allocation_inputs(programs_path, cohorts=cohorts_path)


def test_load_allocation_inputs_optional_parts_may_be_omitted() -> None:
    bundle = load_allocation_inputs(io.StringIO(PROGRAMS_CSV))

    assert bundle.cohorts == ()
    assert bundle.region_floors == ()


def test_load_programs_from_missing_path_reports_error(tmp_path: Path) -> None:
    with pytest.raises(CsvLoaderError, match="cannot open"):
        load_programs(tmp_path / "does-not-exist.csv")


