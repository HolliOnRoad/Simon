#!/bin/zsh
set -e

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
  if [ -x "/opt/homebrew/bin/python3.12" ]; then
    /opt/homebrew/bin/python3.12 -m venv .venv
  elif [ -x "/usr/local/bin/python3.12" ]; then
    /usr/local/bin/python3.12 -m venv .venv
  else
    /usr/bin/env python3 -m venv .venv
  fi
fi

source .venv/bin/activate
pip install -r requirements.txt
pip install pyinstaller pillow

mkdir -p build

python - <<'PY'
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import os

out = Path('build/simon.png')
size = 1024
bg = (15, 23, 42)
accent = (46, 204, 113)
text_color = (255, 255, 255)

img = Image.new("RGBA", (size, size), bg + (255,))
draw = ImageDraw.Draw(img)

pad = 80
radius = 220
draw.rounded_rectangle(
    [pad, pad, size - pad, size - pad],
    radius=radius,
    fill=(20, 30, 50, 255),
    outline=accent + (255,),
    width=18,
)

font_paths = [
    "/System/Library/Fonts/SFNSDisplay.ttf",
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/SFNSRounded.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Helvetica.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]

font = None
for path in font_paths:
    if os.path.exists(path):
        try:
            font = ImageFont.truetype(path, 640)
            break
        except Exception:
            continue
if font is None:
    font = ImageFont.load_default()

text = "S"
bbox = draw.textbbox((0, 0), text, font=font)
tw = bbox[2] - bbox[0]
th = bbox[3] - bbox[1]
x = (size - tw) / 2 - bbox[0]
y = (size - th) / 2 - bbox[1]
draw.text((x, y), text, font=font, fill=text_color + (255,))

img.save(out)
print("icon png written", out)
PY

ICONSET="build/Simon.iconset"
rm -rf "$ICONSET"
mkdir -p "$ICONSET"

sips -z 16 16     build/simon.png --out "$ICONSET/icon_16x16.png" >/dev/null
sips -z 32 32     build/simon.png --out "$ICONSET/icon_16x16@2x.png" >/dev/null
sips -z 32 32     build/simon.png --out "$ICONSET/icon_32x32.png" >/dev/null
sips -z 64 64     build/simon.png --out "$ICONSET/icon_32x32@2x.png" >/dev/null
sips -z 128 128   build/simon.png --out "$ICONSET/icon_128x128.png" >/dev/null
sips -z 256 256   build/simon.png --out "$ICONSET/icon_128x128@2x.png" >/dev/null
sips -z 256 256   build/simon.png --out "$ICONSET/icon_256x256.png" >/dev/null
sips -z 512 512   build/simon.png --out "$ICONSET/icon_256x256@2x.png" >/dev/null
sips -z 512 512   build/simon.png --out "$ICONSET/icon_512x512.png" >/dev/null
sips -z 1024 1024 build/simon.png --out "$ICONSET/icon_512x512@2x.png" >/dev/null

iconutil -c icns "$ICONSET" -o build/Simon.icns

BUNDLE_ID=${BUNDLE_ID:-com.holger.simon}

pyinstaller --noconfirm --windowed --name Simon \
  --icon build/Simon.icns \
  --osx-bundle-identifier "$BUNDLE_ID" \
  --collect-all faster_whisper \
  --collect-all ctranslate2 \
  --collect-all sounddevice \
  --collect-all PySide6 \
  --hidden-import sounddevice \
  --hidden-import faster_whisper \
  --hidden-import ctranslate2 \
  simon.py

rm -rf "Simon.app"
cp -R "dist/Simon.app" "Simon.app"

echo "Built: $SCRIPT_DIR/Simon.app"
