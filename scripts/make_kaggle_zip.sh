#!/usr/bin/env bash
# Zip the committed code (without results/ and docs/) for upload as a private Kaggle dataset; scripts/kaggle_run.sh
# runs it there. The zip holds rasp/ plus rasp/GIT_COMMIT, so every result can be traced to its commit.
#
#   bash scripts/make_kaggle_zip.sh [out.zip]        # default rasp_code.zip
set -euo pipefail
cd "$(dirname "$0")/.."
out=${1:-rasp_code.zip}
git diff --quiet HEAD || echo "warning: uncommitted changes are NOT in the zip (it is made from HEAD)"
git archive --format=zip --prefix=rasp/ -o "$out" HEAD ':!results' ':!docs'
python3 - "$out" "$(git rev-parse --short HEAD)" <<'PY'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1], "a") as z:
    z.writestr("rasp/GIT_COMMIT", sys.argv[2] + "\n")
PY
echo "wrote $out ($(du -h "$out" | cut -f1)), code $(git rev-parse --short HEAD)"
