"""
Download the Tier 1 datasheet archives from Zenodo (record 4558344).

The archives are big, so this resumes partial downloads and checks each file
against the MD5 that Zenodo publishes before calling it done. Re-running it is
safe: anything already downloaded and verified is skipped.

    python scripts/download_corpus.py                  # everything we use
    python scripts/download_corpus.py transistor       # just one archive
"""

import hashlib
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag.settings import ARCHIVE_DIR

RECORD = "https://zenodo.org/api/records/4558344/files"

# name -> md5 as listed on the Zenodo record.
# mouser_opamps is skipped on purpose: it's a smaller scrape that overlaps
# with opamp_dataset, and we only need a sample of op-amps anyway.
ARCHIVES = {
    "transistor_dataset.tar.xz": "f5719a2c397edddf5244e6f5bf0188a4",
    "opamp_dataset.tar.xz": "ada80c05af12c42c4bfd97890cfd78a9",
    "circular_connector_dataset.tar.xz": "ca691bad4818cdde6d005dd0b43c3cca",
}

CHUNK = 1 << 20  # 1 MB


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def download(name: str, expected_md5: str) -> None:
    target = ARCHIVE_DIR / name
    done_marker = target.with_suffix(target.suffix + ".ok")

    if done_marker.exists():
        print(f"{name}: already downloaded and verified")
        return

    # Pick up where we left off if a previous run was interrupted.
    have = target.stat().st_size if target.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}

    with requests.get(f"{RECORD}/{name}/content", headers=headers,
                      stream=True, timeout=60) as r:
        if r.status_code == 416:
            pass  # server says we already have the whole file
        else:
            r.raise_for_status()
            # 200 means the server ignored our Range header, so start over.
            mode = "ab" if r.status_code == 206 else "wb"
            total = have + int(r.headers.get("Content-Length", 0))
            got = have if mode == "ab" else 0
            with open(target, mode) as f:
                for block in r.iter_content(CHUNK):
                    f.write(block)
                    got += len(block)
                    if got % (100 * CHUNK) < CHUNK:
                        print(f"{name}: {got / 1e6:,.0f} / {total / 1e6:,.0f} MB",
                              flush=True)

    print(f"{name}: checking md5...")
    actual = md5_of(target)
    if actual != expected_md5:
        target.unlink()
        raise SystemExit(f"{name}: md5 mismatch ({actual}), deleted - run again")
    done_marker.touch()
    print(f"{name}: ok")


def main() -> None:
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    wanted = sys.argv[1:]
    for name, md5 in ARCHIVES.items():
        if wanted and not any(w in name for w in wanted):
            continue
        download(name, md5)


if __name__ == "__main__":
    main()
