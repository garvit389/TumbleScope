
from pathlib import Path
import csv
import re
import numpy as np

folder = Path("data/sdlcd_raw")
output = Path("docs/sdlcd_audit.csv")
rows = []

if not any(folder.glob("*_DATA.txt")):
    raise SystemExit("No SDLCD source files found; existing audit preserved.")

reserved_periods = {5000, 6000, 7000, 9999}

for file in sorted(folder.glob("*_DATA.txt")):
    times = []
    reported_period = None
    header_points = None

    with open(file, encoding="utf-8-sig") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        if not line:
            continue

        if line.startswith("#"):
            # Header statistics follow the Per[s] heading.
            if "Per[s]" in line and "Points" in line:
                continue

            # Extract period and point count from numeric header.
            if re.match(r"^#\s*[-+]?\d", line):
                values = line[1:].split()
                if len(values) >= 5:
                    try:
                        reported_period = float(values[0])
                        header_points = int(values[4])
                    except ValueError:
                        pass
            continue

        values = line.split()

        try:
            times.append(float(values[0]))
        except (ValueError, IndexError):
            continue

    if len(times) < 2:
        print(f"Skipping {file.name}: insufficient data")
        continue

    times = np.sort(np.array(times))
    intervals = np.diff(times)

    if reported_period in reserved_periods:
        reported_period = None

    rows.append({
        "filename": file.name,
        "object_id": file.name.split("_")[0],
        "points": len(times),
        "header_points": header_points,
        "duration_s": round(float(times[-1] - times[0]), 3),
        "median_cadence_s": round(float(np.median(intervals[intervals > 0])), 4),
        "reference_period_s": reported_period,
    })


# Additional data-quality checks
print("\n--- DATA QUALITY REPORT ---")

for file in sorted(folder.glob("*_DATA.txt")):
    measurements = []
    malformed_rows = 0

    with open(file, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            values = line.split()

            if len(values) != 6:
                malformed_rows += 1
                continue

            try:
                measurements.append(
                    [float(value) for value in values]
                )
            except ValueError:
                malformed_rows += 1

    if not measurements:
        print(f"{file.name}: no valid measurements")
        continue

    data = np.array(measurements)
    times = data[:, 0]

    invalid_rows = int(
        np.sum(~np.all(np.isfinite(data), axis=1))
    )

    intervals = np.diff(np.sort(times))
    positive_intervals = intervals[
        np.isfinite(intervals) & (intervals > 0)
    ]

    if len(positive_intervals):
        median_dt = np.median(positive_intervals)
        large_gaps = int(
            np.sum(positive_intervals > 3 * median_dt)
        )
    else:
        large_gaps = 0

    print(
        f"{file.name}: "
        f"malformed={malformed_rows}, "
        f"invalid={invalid_rows}, "
        f"large_gaps={large_gaps}"
    )


output.parent.mkdir(parents=True, exist_ok=True)

with open(output, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "filename",
        "object_id",
        "points",
        "header_points",
        "duration_s",
        "median_cadence_s",
        "reference_period_s",
    ])
    writer.writeheader()
    writer.writerows(rows)

print(f"Processed {len(rows)} lightcurves")
print(f"Saved report to {output}")
