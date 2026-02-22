#!/bin/zsh
set -e

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d "Simon.app" ]; then
  echo "Simon.app not found. Build it first with ./build_app.sh"
  exit 1
fi

VERSION=$(python3 - <<'PY'
import re
from pathlib import Path
text = Path('simon.py').read_text()
m = re.search(r"APP_VERSION\s*=\s*\"([^\"]+)\"", text)
print(m.group(1) if m else "0.0.0")
PY
)

TAG=${TAG:-"v$VERSION"}
GITHUB_REPO=${GITHUB_REPO:-"HolliOnRoad/Simon"}

mkdir -p dist updates docs/updates

ZIP_PATH="dist/Simon.app.zip"
DMG_PATH="dist/Simon.dmg"
rm -f "$ZIP_PATH" "$DMG_PATH"

ditto -c -k --keepParent "Simon.app" "$ZIP_PATH"

hdiutil create -volname "Simon" -srcfolder "Simon.app" -ov -format UDZO "$DMG_PATH" >/dev/null

URL=${UPDATE_URL:-"https://github.com/$GITHUB_REPO/releases/download/$TAG/Simon.app.zip"}

cat <<JSON > updates/simon.json
{"version":"$VERSION","url":"$URL"}
JSON

cp -f updates/simon.json docs/updates/simon.json

echo "Release-Artefakte bereit:"
echo "- $ZIP_PATH"
echo "- $DMG_PATH"
echo "- updates/simon.json"
echo "- docs/updates/simon.json"

echo "\nNaechste Schritte:"
echo "1) GitHub Release $TAG erstellen und $ZIP_PATH + $DMG_PATH hochladen"
echo "2) updates/simon.json committen (oder via GitHub Pages ausliefern)"
