#!/usr/bin/env bash
# Zips for two private Kaggle datasets that scripts/kaggle_run.sh uses:
#   bash scripts/make_kaggle_zip.sh [out.zip]                # the committed code without results/ and docs/, plus
#                                                            #   rasp/GIT_COMMIT (default rasp_code.zip)
#   bash scripts/make_kaggle_zip.sh --matrices [out.zip]     # data/seq: the 6 checkpoint matrices and their manifest
#                                                            #   (default rasp_matrices.zip, ~100 MB)
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "${1:-}" = --matrices ]; then
  out=${2:-rasp_matrices.zip}
  python3 - "$out" <<'PY'
import pathlib, sys, zipfile
files = sorted(pathlib.Path("data/seq").rglob("*.mtx")) + [pathlib.Path("data/seq/manifest.csv")]
assert len(files) == 7, f"expected 6 matrices and a manifest in data/seq, found {len(files) - 1} matrices"
with zipfile.ZipFile(sys.argv[1], "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for f in files:
        z.write(f, f.relative_to("data"))
PY
  echo "wrote $out ($(du -h "$out" | cut -f1))"
  exit 0
fi
out=${1:-rasp_code.zip}
git diff --quiet HEAD || echo "warning: uncommitted changes are NOT in the zip (it is made from HEAD)"
git archive --format=zip --prefix=rasp/ -o "$out" HEAD ':!results' ':!docs'
python3 - "$out" "$(git rev-parse --short HEAD)" <<'PY'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1], "a") as z:
    z.writestr("rasp/GIT_COMMIT", sys.argv[2] + "\n")
PY
echo "wrote $out ($(du -h "$out" | cut -f1)), code $(git rev-parse --short HEAD)"
