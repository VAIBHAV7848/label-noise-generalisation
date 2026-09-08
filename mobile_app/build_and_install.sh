#!/usr/bin/env bash
set -e

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK_DIR="/home/nethunter/android-sdk"
BUILD_TOOLS="$SDK_DIR/build-tools/34.0.0"
PLATFORM="$SDK_DIR/platforms/android-34/android.jar"

AAPT2="$BUILD_TOOLS/aapt2"
D8="$BUILD_TOOLS/d8"
ZIPALIGN="$BUILD_TOOLS/zipalign"
APKSIGNER="$BUILD_TOOLS/apksigner"

BUILD_TMP="$APP_DIR/build"
mkdir -p "$BUILD_TMP/compiled_res" "$BUILD_TMP/gen" "$BUILD_TMP/classes"

echo "=== 1. Compiling Android Resources with aapt2 ==="
"$AAPT2" compile --dir "$APP_DIR/res" -o "$BUILD_TMP/compiled_res"

echo "=== 2. Linking Resources & Generating R.java ==="
"$AAPT2" link -o "$BUILD_TMP/unaligned.apk" \
    -I "$PLATFORM" \
    --manifest "$APP_DIR/AndroidManifest.xml" \
    --java "$BUILD_TMP/gen" \
    --auto-add-overlay \
    "$BUILD_TMP"/compiled_res/*.flat

echo "=== 3. Compiling Java Source Files ==="
javac -d "$BUILD_TMP/classes" \
    -cp "$PLATFORM:$BUILD_TMP/gen" \
    "$BUILD_TMP/gen/com/research/pilotcontroller/R.java" \
    "$APP_DIR/src/com/research/pilotcontroller/MainActivity.java"

echo "=== 4. Converting Bytecode to DEX with d8 ==="
"$D8" --output "$BUILD_TMP" \
    --lib "$PLATFORM" \
    "$BUILD_TMP"/classes/com/research/pilotcontroller/*.class

echo "=== 5. Adding classes.dex to APK ==="
cd "$BUILD_TMP"
zip -u unaligned.apk classes.dex
cd "$APP_DIR"

echo "=== 6. Zipalign APK ==="
"$ZIPALIGN" -f -v 4 "$BUILD_TMP/unaligned.apk" "$BUILD_TMP/aligned.apk"

echo "=== 7. Signing APK with Persistent Keystore ==="
PERSISTENT_KEYSTORE="$APP_DIR/pilot_debug.keystore"
if [ ! -f "$PERSISTENT_KEYSTORE" ]; then
    keytool -genkey -v -keystore "$PERSISTENT_KEYSTORE" -storepass android -alias androiddebugkey -keypass android -keyalg RSA -keysize 2048 -validity 10000 -dname "CN=Android Debug,O=Android,C=US"
fi

"$APKSIGNER" sign --ks "$PERSISTENT_KEYSTORE" --ks-pass pass:android --key-pass pass:android --out "$BUILD_TMP/PilotController.apk" "$BUILD_TMP/aligned.apk"

echo "=== 8. Installing APK to Connected Android Device via adb ==="
# Uninstall old signature package to prevent signature collision
adb uninstall com.research.pilotcontroller || true
adb install -r "$BUILD_TMP/PilotController.apk"

echo "=== 9. Launching Pilot Controller App on Phone ==="
adb shell am start -n com.research.pilotcontroller/.MainActivity

echo "=== APP INSTALLED AND RUNNING SUCCESSFULLY ON YOUR PHONE ==="
