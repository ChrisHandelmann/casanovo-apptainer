#!/usr/bin/env bash
# Submit Casanovo as a Slurm array: one task (one GPU) per .mgf file.
# Works on any Slurm cluster with Apptainer (CCR, Empire AI) no CCR paths inside.
#
# Usage:
#   ./submit_casanovo.sh <image.sif> <mgf_dir> <out_dir> [sbatch options...]
#
# Results go to <out_dir>/Casanovo_BW-<beam>/<name>.mztab, the same layout
# runCasanovo.sh uses, so the downstream steps don't change.
#
# Optional environment variables:
#   BEAM_SIZE=5        Casanovo beam width
#   CPUS=8  MEM=64G    per-task resources
#   MAX_PARALLEL=8     array tasks running at once
#
# Examples:
#   ./submit_casanovo.sh casanovo-4.3.0-x86_64.sif mgf/ out/ --partition=debug --qos=debug --time=01:00:00
#   ./submit_casanovo.sh casanovo-4.3.0-x86_64.sif mgf/ out/ --partition=eai-test --qos=eai-test --time=12:00:00
set -euo pipefail

if (( $# < 3 )); then
  sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
fi

sif="$(readlink -f "$1")"; mgf_dir="$(readlink -f "$2")"; out_dir="$3"; shift 3
beam="${BEAM_SIZE:-5}"
here="$(dirname "$(readlink -f "$0")")"

[[ -f "$sif" ]]     || { echo "ERROR: image not found: $sif" >&2; exit 1; }
[[ -d "$mgf_dir" ]] || { echo "ERROR: MGF folder not found: $mgf_dir" >&2; exit 1; }
mkdir -p "$out_dir/logs"
out_dir="$(readlink -f "$out_dir")"

shopt -s nullglob
mgfs=("$mgf_dir"/*.mgf)
n=${#mgfs[@]}
(( n > 0 )) || { echo "ERROR: no .mgf files in $mgf_dir" >&2; exit 1; }

echo "[submit] $n MGF files | beam $beam | image $(basename "$sif")"
echo "[submit] results -> $out_dir/Casanovo_BW-$beam"

# Later options win, so anything passed on the command line overrides these.
sbatch \
  --job-name="casanovo_bw${beam}" \
  --array="0-$((n - 1))%${MAX_PARALLEL:-8}" \
  --nodes=1 --ntasks=1 --gpus-per-node=1 \
  --cpus-per-task="${CPUS:-8}" --mem="${MEM:-64G}" \
  --requeue \
  --output="$out_dir/logs/casanovo_%A_%a.out" \
  --export=ALL,CASANOVO_SIF="$sif",MGF_DIR="$mgf_dir",OUT_DIR="$out_dir",BEAM_SIZE="$beam" \
  "$@" \
  "$here/casanovo_task.sbatch"
