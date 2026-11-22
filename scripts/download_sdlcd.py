"""Download the existing audit inventory from the official SDLCD DR3 catalogue.

Local research use was approved by the user. Redistribution remains unapproved.
"""
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
from io import StringIO
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://www.sdlcd.space-debris.sk/"
CATALOGUE = BASE + "static/data/database.json"


def main():
    destination = ROOT / "data/sdlcd_raw"
    destination.mkdir(parents=True, exist_ok=True)
    with urlopen(CATALOGUE, timeout=60) as response:
        catalogue_bytes = response.read()
    catalogue = {row["data-file"]: row for row in json.loads(catalogue_bytes).values()}
    with (ROOT / "docs/sdlcd_audit.csv").open(encoding="utf-8-sig", newline="") as source:
        names = [row["filename"] for row in csv.DictReader(source)]
    if not 20 <= len(names) <= 50 or len(set(names)) != len(names):
        raise ValueError("expected 20 to 50 unique audited files")

    def download(name):
        if Path(name).name != name or name not in catalogue:
            raise ValueError(f"file missing from official catalogue: {name}")
        entry = catalogue[name]
        url = BASE + "static/data/" + name
        path = destination / name
        if path.exists():
            data = path.read_bytes()
        else:
            with urlopen(url, timeout=60) as response:
                data = response.read()
        text = data.decode("utf-8-sig")
        rows = np.loadtxt(StringIO(text), comments="#", ndmin=2)
        if rows.shape != (int(entry["points"]), 6) or not np.isfinite(rows).all():
            raise ValueError(f"invalid measurement layout or count: {name}")
        if name.removesuffix("_DATA.txt") not in text.splitlines()[0]:
            raise ValueError(f"observation identity mismatch: {name}")
        if not path.exists():
            with path.open("xb") as target:
                target.write(data)
        period = float(entry["period"])
        record = {"filename": name, "object_id": entry["object-id"], "source_url": url,
                  "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
                  "catalogue_period_field": period,
                  "reference_period_s": period if period > 0 and period not in (5000,6000,7000,9999) else None,
                  "points": len(rows), "catalogue_entry": entry}
        print(name, len(rows), flush=True)
        return record

    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(download, names))
    manifest = {"release": "SDLCD Data Release 3", "catalogue_url": CATALOGUE,
                "catalogue_sha256": hashlib.sha256(catalogue_bytes).hexdigest(),
                "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                "terms_url": BASE,
                "citation_url": "https://www.sciencedirect.com/science/article/pii/S0273117720300727",
                "use": "User approved local research subject to citation and acknowledgment; redistribution unapproved.",
                "records": records}
    with (destination / "download_manifest.json").open("x", encoding="utf-8") as target:
        json.dump(manifest, target, indent=2, allow_nan=False)
    print("Downloaded and verified", len(records), "observations")


if __name__ == "__main__":
    main()
