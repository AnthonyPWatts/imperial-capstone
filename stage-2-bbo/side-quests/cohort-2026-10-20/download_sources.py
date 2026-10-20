"""Fetch the pinned public data sources, checking every downloaded checksum."""

import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request


STUDY = Path(__file__).resolve().parent
CACHE = STUDY.parents[2] / ".runtime/cohort-side-quest"


def fetch(source):
    target = CACHE / "sources" / source["repository"] / source["path"]
    if target.is_file():
        content = target.read_bytes()
    else:
        request = urllib.request.Request(source["url"], headers={"User-Agent": "Imperial-cohort-study"})
        with urllib.request.urlopen(request, timeout=40) as response:
            content = response.read()
    if hashlib.sha256(content).hexdigest() != source["sha256"]:
        raise ValueError(f"Source checksum mismatch: {source['repository']}/{source['path']}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return {**source, "local_path": str(target)}


if __name__ == "__main__":
    manifest = json.loads((STUDY / "source-manifest.json").read_text(encoding="utf-8"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        sources = list(executor.map(fetch, manifest["sources"]))
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "downloads.json").write_text(json.dumps(sources, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(sources)} pinned public source files.")
