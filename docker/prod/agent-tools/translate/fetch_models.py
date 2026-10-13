#!/usr/bin/env python3
"""Download Argos Translate models for mcp-translate into DIR/<from>_<to>/.

Run on the host (stdlib only), once and after changing LANGUAGES:
    python3 fetch_models.py /mnt/external/argos-models

Each language gets both directions with English (~70-90 MB each). Pairs
already present are skipped; the stanza sentence splitters inside the
packages are not extracted (translate.py splits with a regex).
"""

import io
import json
import os
import shutil
import sys
import urllib.request
import zipfile

INDEX = "https://raw.githubusercontent.com/argosopentech/argospm-index/main/index.json"
# 30 languages with English; keep in step with translate.NAMES.
LANGUAGES = (
    "ar bg ca cs da de el es fa fi fr he hi hu id it ja ko nl pl pt ro ru sk sv th tr uk vi zh"
).split()


def fetch(url: str, timeout: int) -> bytes:
    # argos-net.com answers 403 to urllib's default User-Agent.
    req = urllib.request.Request(url, headers={"User-Agent": "fetch_models/1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def main(target: str) -> None:
    os.makedirs(target, exist_ok=True)
    index = json.loads(fetch(INDEX, 60))
    wanted = {("en", l) for l in LANGUAGES} | {(l, "en") for l in LANGUAGES}
    found = {(p["from_code"], p["to_code"]): p for p in index if (p["from_code"], p["to_code"]) in wanted}
    missing = sorted(wanted - found.keys())
    if missing:
        print("not in the index:", missing, file=sys.stderr)
    for (src, dst), pkg in sorted(found.items()):
        out = os.path.join(target, f"{src}_{dst}")
        if os.path.isfile(os.path.join(out, "model", "model.bin")):
            continue
        url = pkg["links"][0]
        print(f"{src}->{dst} {pkg['package_version']} {url}", flush=True)
        data = fetch(url, 600)
        tmp = out + ".tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for info in z.infolist():
                # translate-el_en-1_9/model/model.bin -> model/model.bin
                rel = info.filename.split("/", 1)[1] if "/" in info.filename else ""
                if not rel or info.is_dir() or rel.startswith("stanza/"):
                    continue
                path = os.path.join(tmp, rel)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with z.open(info) as f, open(path, "wb") as g:
                    shutil.copyfileobj(f, g)
        shutil.rmtree(out, ignore_errors=True)
        os.replace(tmp, out)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
