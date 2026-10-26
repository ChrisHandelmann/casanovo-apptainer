# Casanovo apptainer container

Runs Casanovo in an Apptainer image. Planned to be integrated as a piece into a full pipeline on CCR:
```
CCR                                   EmpireAI (or ANY GPU)      CCR
raw -> MGF (ThermoRawFileParser) ----> Casanovo -> .mztab  ----> NovoLign, LCA, R summaries
```

The apptainer image holds NVIDIA's PyTorch build, Casanovo, and the model weights. 
It reads your MGFs and writes mzTabs into the same `Casanovo_BW-<beam>/` folder layout as `runCasanovo.sh`.

| File | What it does |
|---|---|
| `casanovo.def` | The recipe for the image |
| `casanovo-pins.txt` | Version limits used during the build (optionally your conda env's exact versions) |
| `build_casanovo_sif.sbatch` | Builds the image on a CCR compute node |
| `submit_casanovo.sh` | Submits one GPU task per MGF file, on any Slurm cluster |
| `casanovo_task.sbatch` | What each task runs (called by `submit_casanovo.sh`) |

## 0. Check which Casanovo your pipeline uses

```
casanovo version
```

The image defaults to 4.3.0 with the v4.2.0 weights. 
If yours prints 5.x, build with the matching version (see the top of `build_casanovo_sif.sbatch`).


## 1. Build the image (on CCR)
Specify the image based on desired CPU type:

## x86_64: CCR GPUs, eai-test, Empire AI Alpha
```
sbatch build_casanovo_sif.sbatch 
'''

## aarch64: CCR arm64, Empire AI Beta
```
sbatch --partition=arm64 --qos=arm64 --constraint=ARM64 build_casanovo_sif.sbatch
```

Each build produces `casanovo-<version>-<arch>.sif`, roughly 10–15 GB. Set `SIF_DIR=/desired/folder` for alterante locations.
After the build is complete, the image will launch a test using two test spectra to test if its viable. 
The log `slurm-build-casanovo-<jobid>.out` should end with `Container test passed.`


## 2. Testing on CCR
The apptainer uses the same test mgf as Casanovo itself. Stored in test_mgf dir.
```
./submit_casanovo.sh casanovo-4.3.0-x86_64.sif /path/to/mgf_dir /path/to/test_out \
    --partition=debug --qos=debug --time=01:00:00
```

Logs are written to `test_out/logs/`. Each task also writes `<name>.gpu_usage.csv` (GPU use every 30 s) next to its mzTab. 
Use `jobstats <jobid>` and `ccr-jobview-url <jobid> ub-hpc` to get the efficiency numbers and Grafana links.
Different GPUs and PyTorch builds round slightly differently, which can flip near-tied predictions.


## 3. Run at scale
On CCR's eai-test (12hr limit; check for desired QOS name with `slimits`):

```bash
./submit_casanovo.sh casanovo-4.3.0-x86_64.sif $MGF_DIR $CASANOVO_OUTDIR --partition=eai-test --qos=eai-test --time=12:00:00
```
Preempted tasks requeue automatically. Finished files are skipped, and unfinished are named `<name>.partial.mztab`, so half-written looks incomplete.

For Empire AI:
1. Copy this folder, the matching `.sif` (x86_64 for Alpha, aarch64 for Beta) and your MGF folder over with Globus.
2. Run `submit_casanovo.sh` there, with Empire AI's partition/account options instead of CCR's.
3. Copy `<out_dir>/Casanovo_BW-<beam>/` back to CCR and continue with analysis.

Other settings: `BEAM_SIZE=10 ./submit_casanovo.sh ...` changes the beam (batch size drops to 16 at beam ≥ 8, as before). `CPUS`, `MEM` and `MAX_PARALLEL` change resources.
