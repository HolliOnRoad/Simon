#!/bin/zsh
set -e

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

APP="$SCRIPT_DIR/Simon.app"
ZIP="$SCRIPT_DIR/dist/Simon.app.zip"

if [ ! -d "$APP" ]; then
  echo "Simon.app not found. Build it first with ./build_app.sh"
  exit 1
fi

IDENTITY=${DEVELOPER_ID_APP:-""}
if [ -z "$IDENTITY" ]; then
  echo "Set DEVELOPER_ID_APP to your Developer ID Application certificate name."
  exit 1
fi

mkdir -p "$SCRIPT_DIR/dist"

codesign --force --deep --options runtime --sign "$IDENTITY" "$APP"
codesign --verify --deep --strict "$APP"

ditto -c -k --keepParent "$APP" "$ZIP"

echo "Signed app and created: $ZIP"

if [ -n "$NOTARY_PROFILE" ]; then
  xcrun notarytool submit "$ZIP" --keychain-profile "$NOTARY_PROFILE" --wait
  xcrun stapler staple "$APP"
  spctl -a -vv "$APP"
  echo "Notarization complete (profile: $NOTARY_PROFILE)."
  exit 0
fi

if [ -n "$APPLE_ID" ] && [ -n "$APPLE_TEAM_ID" ] && [ -n "$APP_SPECIFIC_PASSWORD" ]; then
  xcrun notarytool submit "$ZIP" --apple-id "$APPLE_ID" --team-id "$APPLE_TEAM_ID" --password "$APP_SPECIFIC_PASSWORD" --wait
  xcrun stapler staple "$APP"
  spctl -a -vv "$APP"
  echo "Notarization complete (apple-id/team-id)."
  exit 0
fi

echo "Notarization skipped. Set NOTARY_PROFILE or APPLE_ID/APPLE_TEAM_ID/APP_SPECIFIC_PASSWORD."
