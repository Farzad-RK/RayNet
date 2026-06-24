#!/usr/bin/env bash
# Tier-3 validation for the on-device 3D eyeball model (EyeModel3D.kt).
#
# Compiles the REAL EyeGeometry.kt + EyeModel3D.kt with Tier3Check.kt on a plain
# JVM (no Android toolchain) and runs them against the synthetic ground-truth
# look-around set from sixdrepnet/3DeepVOG-main/tools/tier3_parity.py.
#
# Validates: two-fold disambiguation + gaze direction (vs ground truth) and the
# eyeball-centre/radius fit (parity vs the reference numpy intersect).
#
#   ./run.sh            # regenerate golden (needs the 3DeepVOG .venv) then check
#   ./run.sh --no-gen   # reuse the existing tier3_golden.txt
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/../../app/src/main/java/com/raynet/eyepatch"
REPO="$HERE/../../../.."
VENV="$REPO/sixdrepnet/3DeepVOG-main/.venv/bin/python"
GOLDEN="$REPO/sixdrepnet/3DeepVOG-main/tools/tier3_golden.txt"

if [[ "${1:-}" != "--no-gen" ]]; then
  echo ">> generating golden via 3DeepVOG reference ..."
  "$VENV" "$REPO/sixdrepnet/3DeepVOG-main/tools/tier3_parity.py"
fi

# Ensure a Kotlin compiler (cached under ~/.cache so it survives /tmp wipes).
if ! command -v kotlinc >/dev/null 2>&1; then
  KHOME="${KOTLINC_HOME:-$HOME/.cache/raynet-kotlinc}"
  if [[ ! -x "$KHOME/kotlinc/bin/kotlinc" ]]; then
    echo ">> downloading kotlin-compiler-2.0.20 to $KHOME ..."
    mkdir -p "$KHOME"
    curl -fsSL -o "$KHOME/kotlinc.zip" \
      https://github.com/JetBrains/kotlin/releases/download/v2.0.20/kotlin-compiler-2.0.20.zip
    unzip -q -o "$KHOME/kotlinc.zip" -d "$KHOME"
  fi
  export PATH="$KHOME/kotlinc/bin:$PATH"
fi

OUT="$(mktemp -d)"
echo ">> compiling EyeGeometry + EyeModel3D + Tier3Check ..."
kotlinc "$SRC/EyeGeometry.kt" "$SRC/EyeModel3D.kt" "$HERE/Tier3Check.kt" \
  -include-runtime -d "$OUT/tier3.jar"
echo ">> running ..."
java -jar "$OUT/tier3.jar" "$GOLDEN"
