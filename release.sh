#!/bin/zsh
set -e

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d "Simon.app" ]; then
  echo "Simon.app not found. Build it first with ./build_app.sh"
  exit 1
fi

VERSION=$(python - <<'PY'
import re
from pathlib import Path
text = Path('simon.py').read_text()
m = re.search(r"APP_VERSION\s*=\s*\"([^\"]+)\"", text)
print(m.group(1) if m else "0.0.0")
PY
)

TAG=${TAG:-"v$VERSION"}
GITHUB_REPO=${GITHUB_REPO:-"holger/simon"}

mkdir -p dist updates

ZIP_PATH="dist/Simon.app.zip"
rm -f "$ZIP_PATH"

ditto -c -k --keepParent "Simon.app" "$ZIP_PATH"

URL=${UPDATE_URL:-"https://github.com/$GITHUB_REPO/releases/download/$TAG/Simon.app.zip"}

cat <<JSON > updates/simon.json
{"version":"$VERSION","url":"$URL"}
JSON

echo "Release artifacts ready:"
echo "- $ZIP_PATH"
echo "- updates/simon.json"

echo "\nNext steps:"
echo "1) Create a GitHub Release $TAG and upload $ZIP_PATH"
echo "2) Commit updates/simon.json to your repo (or host it via GitHub Pages)"
