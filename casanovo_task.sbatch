#!/bin/bash
# One Casanovo array task: sequences the Nth .mgf file in MGF_DIR.
# Submitted by submit_casanovo.sh, which sets CASANOVO_SIF, MGF_DIR, OUT_DIR, BEAM_SIZE.
set -euo pipefail
# Jobs inherit the submitting shell environment. CCR login nodes put x86-only
# tools first on PATH, which breaks on ARM nodes, so system tools go first.
export PATH="/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:/sbin${PATH:+:$PATH}"
unset LD_LIBRARY_PATH LD_PRELOAD

: "${CASANOVO_SIF:?set by submit_casanovo.sh}"
: "${MGF_DIR:?set by submit_casanovo.sh}"
: "${OUT_DIR:?set by submit_casanovo.sh}"
BEAM_SIZE="${BEAM_SIZE:-5}"
TASK="${SLURM_ARRAY_TASK_ID:?not running as a Slurm array task}"

RUN_DIR="$OUT_DIR/Casanovo_BW-${BEAM_SIZE}"
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"   # start where the container can always write (Lightning writes lightning_logs/ here)

shopt -s nullglob
mapfile -t MGF_LIST < <(printf '%s\n' "$MGF_DIR"/*.mgf | sort)
IN="${MGF_LIST[$TASK]:-}"
[[ -n "$IN" && -f "$IN" ]] || { echo "No input for array index $TASK" >&2; exit 1; }
base="$(basename "$IN" .mgf)"
OUT="$RUN_DIR/${base}.mztab"

echo "[info] host $(hostname) ($(uname -m)) | job ${SLURM_JOB_ID:-na} task $TASK | restarts ${SLURM_RESTART_COUNT:-0}"
echo "[info] $IN -> $OUT (beam $BEAM_SIZE)"

# Preemptible partitions cancel and requeue jobs. Finished files are skipped;
# an interrupted run only ever left a .partial file, which is redone.
if [[ -s "$OUT" ]]; then
  echo "[skip] already done"
  exit 0
fi

# -------- container command --------
APPTAINER_BIN="$(command -v apptainer || command -v singularity || true)"
if [[ -z "$APPTAINER_BIN" ]]; then
  module load apptainer 2>/dev/null || true
  APPTAINER_BIN="$(command -v apptainer || command -v singularity || true)"
fi
[[ -n "$APPTAINER_BIN" ]] || { echo "ERROR: apptainer/singularity not found on this node" >&2; exit 1; }

BINDS="$MGF_DIR,$RUN_DIR${EXTRA_BINDS:+,$EXTRA_BINDS}"
ctr() {
  # Host Python/library settings must not leak into the image.
  env -u LD_LIBRARY_PATH -u LD_PRELOAD -u PYTHONPATH -u PYTHONHOME \
    "$APPTAINER_BIN" exec --nv --bind "$BINDS" "$CASANOVO_SIF" "$@"
}

# -------- GPU check (requeue a few times if the node is bad, then give up) --------
requeue_or_fail() {
  if (( ${SLURM_RESTART_COUNT:-0} < ${MAX_GPU_REQUEUES:-3} )); then
    echo "[warn] $1; requeueing"
    scontrol requeue "${SLURM_JOB_ID:?}" || true
    exit 0
  fi
  echo "ERROR: $1 (already requeued ${SLURM_RESTART_COUNT} times)" >&2
  exit 1
}
if ! ctr python3 -c '
import sys, torch
if not torch.cuda.is_available():
    sys.exit("PyTorch cannot see a GPU")
(torch.ones(8, device="cuda") * 2).sum().item()   # run a real kernel
p = torch.cuda.get_device_properties(0)
print(f"[info] GPU {p.name} sm_{p.major}{p.minor} {p.total_memory / 2**30:.0f} GiB | torch {torch.__version__} CUDA {torch.version.cuda}")
'; then
  requeue_or_fail "GPU not usable from the container"
fi

version="$(ctr casanovo version 2>/dev/null | awk -F'[: ]+' '/^Casanovo/{print $2}')"
major="${version%%.*}"
echo "[info] Casanovo $version"

# -------- per-file config: Casanovo defaults + your beam settings --------
CFG="$RUN_DIR/casanovo_${base}.yaml"
ctr cat /opt/casanovo/default_config.yaml > "$CFG"
grep -Eq '^[[:space:]]*residues[[:space:]]*:' "$CFG" || { echo "ERROR: bad default config in image" >&2; exit 1; }

set_key() {   # set_key KEY VALUE: replace the line if present, else append
  if grep -Eq "^[[:space:]]*$1[[:space:]]*:" "$CFG"; then
    sed -E -i "s/^[[:space:]]*$1[[:space:]]*:.*/$1: $2/" "$CFG"
  else
    printf '\n%s: %s\n' "$1" "$2" >> "$CFG"
  fi
}
set_key n_beams "$BEAM_SIZE"
if grep -Eq '^[[:space:]]*beam_size[[:space:]]*:' "$CFG"; then set_key beam_size "$BEAM_SIZE"; fi
if (( BEAM_SIZE >= 8 )); then set_key predict_batch_size 16; fi   # avoid GPU out-of-memory

# -------- GPU utilization log (for judging efficiency; also: jobstats <jobid> on CCR) --------
GPU_LOG_PID=""
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total \
    --format=csv -l "${GPU_LOG_INTERVAL:-30}" > "$RUN_DIR/${base}.gpu_usage.csv" 2>/dev/null &
  GPU_LOG_PID=$!
fi
trap '[[ -n "$GPU_LOG_PID" ]] && kill "$GPU_LOG_PID" 2>/dev/null || true' EXIT

# -------- run --------
PARTIAL="${base}.partial"
rm -f "$RUN_DIR/$PARTIAL.mztab" "$RUN_DIR/$PARTIAL.log"
if (( major >= 5 )); then
  ctr casanovo sequence "$IN" --config "$CFG" --model /opt/casanovo/casanovo.ckpt \
    --output_dir "$RUN_DIR" --output_root "$PARTIAL"
else
  ctr casanovo sequence "$IN" --config "$CFG" --model /opt/casanovo/casanovo.ckpt \
    --output "$RUN_DIR/$PARTIAL.mztab"
fi

mv -f "$RUN_DIR/$PARTIAL.mztab" "$OUT"
if [[ -f "$RUN_DIR/$PARTIAL.log" ]]; then mv -f "$RUN_DIR/$PARTIAL.log" "$RUN_DIR/${base}.log"; fi
echo "[done] $OUT ($(grep -c '^PSM' "$OUT" || true) PSMs)"
