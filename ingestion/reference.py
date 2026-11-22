"""Evaluation-only reference labels and deterministic object splits."""
import csv
import hashlib
from pathlib import Path

from core.schemas import ReferenceLabel


RESERVED_PERIOD_CODES = frozenset((5000., 6000., 7000., 9999.))


def load_reference_table(path: str) -> dict[str, ReferenceLabel]:
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not {"obs_id", "object_id", "period_s"} <= set(reader.fieldnames or []):
            raise ValueError("reference table requires obs_id, object_id and period_s")
        rows = list(reader)
    result, assignments = {}, {}
    for row in rows:
        if not row["period_s"].strip():
            continue
        period = float(row["period_s"])
        if period in RESERVED_PERIOD_CODES:
            continue
        obs_id, object_id = row["obs_id"].strip(), row["object_id"].strip()
        if not obs_id or not object_id or obs_id in result:
            raise ValueError("empty identity or duplicate observation in reference table")
        # Fixed seed 0 and the object identity determine membership, independent of row order.
        digest = hashlib.sha256(("0:" + object_id).encode()).digest()
        split = row.get("split", "").strip() or ("val" if digest[0] % 2 == 0 else "test")
        label = ReferenceLabel(obs_id, object_id, period, split)
        label.validate()
        if object_id in assignments and assignments[object_id] != split:
            raise ValueError("object occurs in both reference splits")
        assignments[object_id] = split
        result[obs_id] = label
    print("Reference observations", {s: sum(v.split == s for v in result.values()) for s in ("val", "test")})
    return result


def main():
    """Export the verified audit inventory with evaluation-only object splits."""
    import argparse
    from tempfile import TemporaryDirectory

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.audit.open(encoding="utf-8-sig", newline="") as source:
        audit = list(csv.DictReader(source))
    with TemporaryDirectory() as folder:
        staging = Path(folder) / "references.csv"
        with staging.open("w", encoding="utf-8", newline="") as target:
            writer = csv.writer(target)
            writer.writerow(("obs_id", "object_id", "period_s"))
            for row in audit:
                writer.writerow((Path(row["filename"]).stem, row["object_id"], row["reference_period_s"]))
        labels = load_reference_table(str(staging))
    with args.output.open("x", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(("obs_id", "object_id", "period_s", "split"))
        for label in labels.values():
            writer.writerow((label.obs_id, label.object_id, label.period_s, label.split))


if __name__ == "__main__":
    main()
