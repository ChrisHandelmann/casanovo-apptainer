#!/bin/bash -l
# Build the Casanovo image on CCR. Submit from this folder:
#
#   x86_64  (CCR GPUs, eai-test, Empire AI Alpha):
#     sbatch build_casanovo_sif.sbatch
#
#   aarch64 (CCR arm64/GH200, Empire AI Beta):
#     sbatch --partition=arm64 --qos=arm64 --constraint=ARM64 build_casanovo_sif.sbatch
#
# Different Casanovo version (weights must match its generation):
#   sbatch --export=ALL,CASANOVO_VERSION=5.0.0,CASANOVO_WEIGHTS_URL=https://github.com/Noble-Lab/casanovo/releases/download/v5.0.0/casanovo_v5_0_0.ckpt build_casanovo_sif.sbatch
#
# No GPU needed. Utilizing NVIDIA's base image.

#SBATCH --job-name=build_casanovo_sif
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-build-casanovo-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
[[ -f casanovo.def ]] || { echo "ERROR: submit this from the folder containing casanovo.def" >&2; exit 1; }

version="${CASANOVO_VERSION:-4.3.0}"
arch="$(uname -m)"
sif_dir="${SIF_DIR:-$PWD}"
sif="$sif_dir/casanovo-${version}-${arch}.sif"
mkdir -p "$sif_dir"

args=(--build-arg "CASANOVO_VERSION=$version")
[[ -n "${CASANOVO_WEIGHTS_URL:-}" ]] && args+=(--build-arg "CASANOVO_WEIGHTS_URL=$CASANOVO_WEIGHTS_URL")
[[ -n "${NGC_TAG:-}" ]] && args+=(--build-arg "NGC_TAG=$NGC_TAG")

# Download cache and build scratch on node-local disk (deleted with the job),
# The finished image lands in storage.
export APPTAINER_TMPDIR="${SLURMTMPDIR:-/tmp}"
export APPTAINER_CACHEDIR="$APPTAINER_TMPDIR/apptainer-cache"
mkdir -p "$APPTAINER_CACHEDIR"

echo "[build] $(hostname -s) ($arch) -> $sif"
echo "[build] ${args[*]}"
apptainer --version

# Built under a temporary name 
# A failed build never replaces a working image.
tmp="${sif%.sif}.building-${SLURM_JOB_ID:-$$}.sif"
trap 'rm -f "$tmp"' EXIT
apptainer build --fakeroot "${args[@]}" "$tmp" casanovo.def
mv -f "$tmp" "$sif"
trap - EXIT

echo "[build] done: $sif ($(du -h "$sif" | cut -f1))"
apptainer inspect --labels "$sif"
