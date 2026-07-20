#!/bin/zsh
# Builds a double-clickable Bonsai.app into dist/.
# Bundles the release binary, all trained weights, and an icon generated from
# an actual render of the trained NCA (the app's icon is a picture the network grew).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> building release binary"
swift build -c release

APP=dist/Bonsai.app
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

cp .build/release/Bonsai "$APP/Contents/MacOS/Bonsai"
cp weights/*.nca "$APP/Contents/Resources/" 2>/dev/null || {
    echo "!! no weights in weights/ — train first"; exit 1; }
# statemaps + anchors: the explorer's fallback path loads statemap_2d.json from
# the same directory as the weights — omitting these silently breaks the
# State Space panel for every creature without flagStates.
cp weights/*.json "$APP/Contents/Resources/" 2>/dev/null || true

cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key><string>ai.aureum.bonsai</string>
    <key>CFBundleName</key><string>Bonsai</string>
    <key>CFBundleDisplayName</key><string>Bonsai</string>
    <key>CFBundleExecutable</key><string>Bonsai</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>0.2.0</string>
    <key>CFBundleIconFile</key><string>AppIcon</string>
    <key>LSUIElement</key><true/>
    <key>LSMinimumSystemVersion</key><string>13.0</string>
    <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
PLIST

echo "==> rendering icon from the trained NCA"
ICON_SRC=$(mktemp -d)/icon.png
.build/release/Bonsai --render-test "$ICON_SRC" 400 weights/bonsai.nca
ICONSET=$(mktemp -d)/AppIcon.iconset
mkdir -p "$ICONSET"
for sz in 16 32 64 128 256; do
    sips -z $sz $sz "$ICON_SRC" --out "$ICONSET/icon_${sz}x${sz}.png" >/dev/null
done
cp "$ICONSET/icon_32x32.png"   "$ICONSET/icon_16x16@2x.png"
cp "$ICONSET/icon_64x64.png"   "$ICONSET/icon_32x32@2x.png"
cp "$ICONSET/icon_256x256.png" "$ICONSET/icon_128x128@2x.png"
iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/AppIcon.icns"

codesign --force --sign - "$APP" 2>/dev/null || true
echo "==> done: $APP"
