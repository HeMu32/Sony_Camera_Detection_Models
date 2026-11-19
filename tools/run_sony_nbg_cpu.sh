#!/usr/bin/env bash
# Run unmodified Sony 0xd4 NBGs through the pinned x86-64 SDK.
set -euo pipefail

if (( $# < 4 )); then
    printf 'Usage: bash %s SDK_DIR BUILD_DIR MODEL.nb OUTPUT_PREFIX INPUT.raw [...]\n' "$0" >&2
    exit 2
fi

sdk=$(realpath "$1")
build=$2
model=$3
shift 3
here=$(cd -- "$(dirname -- "$0")" && pwd)

# Avoid applying version-specific interposition to a different emulator ABI.
(
    cd "$sdk/lib"
    printf '%s\n' \
      '56e34c51ca5476e0ca54155d4e78679d96281552ddad7674f5ce4fe2fe5f7de6  libEmulator.so' \
      '423514a8f50c8d3c15d7b257b2027a1cd9434051a74ed0b2936b2df38badeb38  libOpenVX.so.1.3.0' |
      sha256sum -c --status
) || { printf 'Unsupported VeriSilicon SDK build\n' >&2; exit 1; }
python3 - "$model" <<'PY'
import struct
import sys

with open(sys.argv[1], "rb") as graph:
    magic, version, target = struct.unpack("<4sII", graph.read(12))
if magic != b"VPMN" or target != 0xd4 or version not in (0x1000d, 0x1000f, 0x10016):
    raise SystemExit("Not a supported Sony 0xd4 NBG")
PY

mkdir -p -- "$build"
build=$(realpath "$build")
cc -O2 -Wall -Wextra -Werror -std=c11 -I"$sdk/include" "$here/nbg_cpu_probe.c" \
   -L"$sdk/lib" -Wl,-rpath-link,"$sdk/lib" -l:libOpenVX.so.1.3.0 -o "$build/nbg_cpu_probe"
for shim in vsim_nbg_mode vsim_legacy_sram_reloc; do
    cc -O2 -Wall -Wextra -Werror -shared -fPIC "$here/$shim.c" -ldl -o "$build/$shim.so"
done
ln -sfn -- "$sdk/lib/libOpenVX.so.1.3.0" "$build/libOpenVX.so.1"
export LD_LIBRARY_PATH="$build:$sdk/lib"
export VSIMULATOR_CONFIG=VIP9000NANODI_PID0XD4
preload="$build/vsim_nbg_mode.so:$build/vsim_legacy_sram_reloc.so"
if [[ -n ${NBG_COEFF_TRACE:-} ]]; then
    cc -O2 -Wall -Wextra -Werror -shared -fPIC "$here/trace_nbg_coefficients.c" \
       -ldl -o "$build/trace_nbg_coefficients.so"
    preload="$preload:$build/trace_nbg_coefficients.so"
fi
if [[ -n ${NBG_TP_TRACE:-} || -n ${NBG_NN_WRITES:-} ]]; then
    cc -O2 -Wall -Wextra -Werror -shared -fPIC "$here/trace_nbg_tp.c" \
       -ldl -o "$build/trace_nbg_tp.so"
    preload="$preload:$build/trace_nbg_tp.so"
fi
export LD_PRELOAD="$preload"
exec "$build/nbg_cpu_probe" "$model" "$@"
