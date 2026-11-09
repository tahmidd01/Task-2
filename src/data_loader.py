"""Fetch the official archive when needed and verify original CSVs."""

import csv
import hashlib
import io
import json
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

import pandas as pd

SOURCE = "https://www.kaggle.com/datasets/gokulrajkmv/unemployment-in-india"
DOWNLOAD = "https://www.kaggle.com/api/v1/datasets/download/gokulrajkmv/unemployment-in-india"
FILES = {
    "area": "Unemployment in India.csv",
    "regional": "Unemployment_Rate_upto_11_2020.csv",
}


def fingerprint(payload):
    rows = list(csv.reader(io.StringIO(payload.decode("utf-8-sig"))))
    if len(rows) < 2 or any(len(row) != len(rows[0]) for row in rows[1:]):
        raise ValueError("CSV is empty or contains malformed records")
    return {
        "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
        "rows": len(rows) - 1, "columns": len(rows[0]),
    }


def load_raw(folder):
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / "dataset_metadata.json"
    metadata = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else None
    if any(not (folder / name).is_file() for name in FILES.values()):
        try:
            request = Request(DOWNLOAD, headers={"User-Agent": "Unemployment-Analysis/1.0"})
            with urlopen(request, timeout=45) as response:
                payload = response.read()
            with ZipFile(io.BytesIO(payload)) as archive:
                archive_csvs = sorted(name for name in archive.namelist() if name.lower().endswith(".csv"))
                if archive_csvs != sorted(FILES.values()) or archive.testzip() is not None:
                    raise ValueError("Kaggle archive inventory or integrity check failed")
                originals = {name: archive.read(name) for name in archive_csvs}
            downloaded = {name: fingerprint(data) for name, data in originals.items()}
            for name, data in originals.items():
                path = folder / name
                if path.exists() and path.read_bytes() != data:
                    raise ValueError(f"{name} differs from the current Kaggle release. Preserve the old release separately before replacing it.")
                if metadata and downloaded[name] != metadata["files"][name]:
                    raise ValueError(f"{name} differs from the recorded Kaggle release")
            for name, data in originals.items():
                path = folder / name
                if not path.exists():
                    path.write_bytes(data)
            metadata = {
                "source": SOURCE, "download_url": DOWNLOAD,
                "acquisition": "Official Kaggle public dataset API (unauthenticated ZIP)",
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                "archive_sha256": hashlib.sha256(payload).hexdigest(), "files": downloaded,
            }
        except (HTTPError, URLError, TimeoutError, BadZipFile) as error:
            raise RuntimeError(f"Could not download the Kaggle ZIP. Check connectivity or extract the original CSVs from {SOURCE} into {folder}.") from error

    frames = {}
    observed = {}
    for grain, name in FILES.items():
        payload = (folder / name).read_bytes()
        observed[name] = fingerprint(payload)
        if metadata and observed[name] != metadata["files"][name]:
            raise ValueError(f"Integrity check failed for {name}")
        frames[grain] = pd.read_csv(io.BytesIO(payload), skip_blank_lines=False)
    if metadata is None:
        metadata = {"source": SOURCE, "download_url": DOWNLOAD,
                    "acquisition": "Existing local CSVs; acquisition date unavailable", "files": observed}
    if not manifest.exists():
        manifest.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return frames, metadata

