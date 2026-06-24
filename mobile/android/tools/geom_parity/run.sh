#!/usr/bin/env bash
# JVM golden-value validation for the on-device geometry port (Option A).
#
# Compiles app/src/main/java/com/raynet/eyepatch/EyeGeometry.kt together with
# GeomParityCheck.kt and runs it on a plain JVM (no Android toolchain needed),
# comparing EyeGeometry against the Python reference in
# sixdrepnet/3DeepVOG-main/tools/geom_parity.py (+ geom_golden.json).
#
# What it checks (all to ~machine precision unless noted):
#   - ellipseToGeneral      vs convert_ell_to_general
#   - unproject             vs unprojectGazePositions (normals + centers; the
#                              centered/axis-aligned degeneracy returns null)
#   - intersectLines        vs intersect()
#   - lineSphereIntersect   vs line_sphere_intersect()
#   - reprojection frame sanity (perspective offset expected, <2px) — confirms
#                              EyeModel3D.project()'s frame matches unproject
#   - synthetic eyeball-fit round-trip (intersectLines + eyeRadius recover a
#                              known eyeball center + pupil offset)
#
# Requires: a Kotlin compiler on PATH (kotlinc) and a JRE. To get a standalone
# compiler without sudo:
#   curl -L -o kotlinc.zip \
#     https://github.com/JetBrains/kotlin/releases/download/v2.0.20/kotlin-compiler-2.0.20.zip
#   unzip kotlinc.zip -d /tmp/kotlinhome
#   export PATH=/tmp/kotlinhome/kotlinc/bin:$PATH
#
# To regenerate the golden values the expected numbers are pinned against:
#   cd ../../../sixdrepnet/3DeepVOG-main && .venv/bin/python tools/geom_parity.py
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
GEOM="$HERE/../../app/src/main/java/com/raynet/eyepatch/EyeGeometry.kt"
OUT="$(mktemp -d)"
kotlinc "$GEOM" "$HERE/GeomParityCheck.kt" -include-runtime -d "$OUT/geomcheck.jar"
java -jar "$OUT/geomcheck.jar"
