import csv
from collections import Counter
from decimal import Decimal
from pathlib import Path
from statistics import median

import pytest


ROOT = Path(__file__).resolve().parents[2]
RESERVED = {5000: 3, 6000: 4, 7000: 1, 9999: 10}
DUPLICATE_FILE = "15010C_1_2_3_R_20190531_DATA.txt"


@pytest.fixture(scope="module")
def audit_rows():
    with (ROOT / "docs" / "sdlcd_audit.csv").open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


@pytest.fixture(scope="module")
def observations(audit_rows):
    paths = sorted((ROOT / "data" / "sdlcd_raw").glob("*_DATA.txt"))
    if not paths:
        pytest.skip("Local SDLCD raw files are absent from this checkout")
    assert {path.name for path in paths} == {row["filename"] for row in audit_rows}

    result = {}
    for path in paths:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        headers = [line[1:].strip() for line in lines if line.startswith("#")]
        stats_index = next(i for i, line in enumerate(headers) if line.startswith("Per[s]"))
        stats = dict(zip(headers[stats_index].split(), headers[stats_index + 1].split(), strict=True))
        measurements = [
            tuple(map(Decimal, line.split()))
            for line in lines
            if line.strip() and not line.startswith("#")
        ]
        assert headers[0] + "_DATA.txt" == path.name
        assert headers[-1].split() == ["Time", "ADU", "ADU_err", "MAG", "MAG_err"]
        assert all(len(row) == 6 for row in measurements), path.name
        duration_min = Decimal(headers[2].split(":", 1)[1].strip())
        result[path.name] = (stats, duration_min, measurements)
    return result


def test_audit_csv_inventory(audit_rows):
    assert len(audit_rows) == len({row["filename"] for row in audit_rows}) == 47
    assert sum(int(row["points"]) for row in audit_rows) == 10719
    known = [row for row in audit_rows if row["reference_period_s"]]
    assert len(known) == 29
    assert len({row["object_id"] for row in known}) == 25
    assert all(row["points"] == row["header_points"] for row in audit_rows)
    assert all(row["object_id"] == row["filename"].split("_", 1)[0] for row in audit_rows)
    assert all(0 < Decimal(row["reference_period_s"]) not in RESERVED for row in known)


def test_raw_point_counts_and_measurements(observations):
    assert sum(len(rows) for _, _, rows in observations.values()) == 10719
    for filename, (stats, _, rows) in observations.items():
        assert len(rows) == int(stats["Points"]), filename
        assert all(value.is_finite() for row in rows for value in row), filename
        assert all(row[1] == Decimal(stats["Exp[s]"]) for row in rows), filename


def test_raw_timestamps_and_duplicate(observations):
    duplicates = []
    for filename, (_, duration_min, rows) in observations.items():
        times = [row[0] for row in rows]
        assert times[0] == 0, filename
        assert abs(times[-1] - duration_min * 60) <= Decimal("0.006"), filename
        for left, right in zip(rows, rows[1:]):
            assert left[0] <= right[0], filename
            if left[0] == right[0]:
                assert left == right, filename
                duplicates.append((filename, left[0]))
    assert duplicates == [(DUPLICATE_FILE, Decimal("368.064"))]


def test_raw_period_field_and_reserved_codes(observations, audit_rows):
    codes = Counter()
    for row in audit_rows:
        stats, _, _ = observations[row["filename"]]
        period = Decimal(stats["Per[s]"])
        if period in RESERVED:
            codes[period] += 1
            assert row["reference_period_s"] == "", row["filename"]
        else:
            assert period > 0
            assert Decimal(row["reference_period_s"]) == period, row["filename"]
    assert codes == RESERVED


def test_audit_csv_matches_raw_counts_duration_and_cadence(observations, audit_rows):
    for row in audit_rows:
        filename = row["filename"]
        stats, _, measurements = observations[filename]
        times = sorted(item[0] for item in measurements)
        positive_intervals = [right - left for left, right in zip(times, times[1:]) if right > left]
        assert int(row["points"]) == len(times), filename
        assert int(row["header_points"]) == int(stats["Points"]), filename
        assert Decimal(row["duration_s"]) == times[-1] - times[0], filename
        assert Decimal(row["median_cadence_s"]) == median(positive_intervals), filename
