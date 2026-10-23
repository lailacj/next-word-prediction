# Running on Brown's Oscar cluster

## Project and environment

The runtime copy is at `/oscar/data/rfeiman/jdehaan/next-word-prediction`.
It contains the maintained code, tests, documentation, and four prepared datasets.
Imported source collections and legacy results remain in the local repository.
`SOURCE_COMMIT` identifies the commit used for the runtime copy; each score's
metadata also records hashes of the Python source files.

In each new SSH session, activate the environment before submitting jobs:

```sh
ssh <username>@ssh.ccv.brown.edu
cd /oscar/data/rfeiman/jdehaan/next-word-prediction
module load python/3.12.8-zr3c
source .venv/bin/activate
export HF_HUB_CACHE="$PWD/.cache/huggingface/hub"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
```

Model files are cached in project storage. Authentication uses Hugging Face's
default location in your home directory; keep this distinction when setting up
jobs. Do not copy the Mac's virtual environment to Oscar.

### Recreating the environment

The L40S node checked during setup supports CUDA 12.9. The default PyPI Torch
wheel used CUDA 13 and failed device initialization. Use the official CUDA 12.8
wheel instead. In a fresh virtual environment, install Torch first, then the
project:

```sh
python -m pip install 'torch==2.11.0+cu128' --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
python -m pip check
```

This wheel is listed in the [official PyTorch CUDA 12.8 index](https://download.pytorch.org/whl/cu128/torch/).
Check CUDA in an allocated GPU job after installation; a successful package
install alone does not establish driver compatibility.

## Enable Llama access

1. Sign in to Hugging Face and request access to
   [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B).
   Review and accept the model's access conditions yourself. Wait for access to
   be granted to your account.
2. In [token settings](https://huggingface.co/settings/tokens), create a token
   for Oscar with read access to this gated model. A fine-grained token limits
   its scope; write access is unnecessary. See the
   [Hugging Face token guide](https://huggingface.co/docs/hub/security-tokens).
3. In the activated Oscar environment, run:

   ```sh
   hf auth login
   hf auth whoami
   hf download meta-llama/Llama-3.2-1B config.json
   ```

   Follow the interactive login prompt (choose the token option if offered).
   Paste the token only into the terminal prompt. If asked to add it as a Git
   credential, answer no. The final command checks model access with a small
   configuration download; successful login alone does not prove gated access.
   See the [CLI guide](https://huggingface.co/docs/huggingface_hub/guides/cli).

The pipeline automatically uses the saved login. No token belongs in a job
script or Git. BERT, DeepSeek, and Qwen can run while Llama access
is pending.

## Pilot submissions

Submit from the project root after activation. Oscar runs inference on allocated
compute nodes; see its [job guide](https://docs.ccv.brown.edu/oscar/submitting-jobs/shared-machine).
The following requests an L40S GPU explicitly so the hardware choice is stable
across pilots. The account currently has access through partition `gpu`.

```sh
sbatch --partition=gpu --gres=gpu:l40s:1 --cpus-per-task=4 --mem=64G --time=01:00:00 \
  scripts/run_pipeline.sh --dataset data/processed/michaelov_2024.csv \
  --model bert --device cuda --dtype float32 \
  --output results/pilot/michaelov_2024/bert.csv

sbatch --partition=gpu --gres=gpu:l40s:1 --cpus-per-task=4 --mem=64G --time=01:00:00 \
  scripts/run_pipeline.sh --dataset data/processed/michaelov_2024.csv \
  --model deepseek --device cuda --dtype float32 \
  --output results/pilot/michaelov_2024/deepseek.csv

sbatch --partition=gpu --gres=gpu:l40s:1 --cpus-per-task=4 --mem=64G --time=01:00:00 \
  scripts/run_pipeline.sh --dataset data/processed/michaelov_2024.csv \
  --model llama --device cuda --dtype float32 \
  --revision 4e20de362430cd3b72f300e6b0f18e50e7166e08 \
  --output results/pilot/michaelov_2024/llama.csv
```

These commands inherit the active environment. SLURM command-line options
override the launcher's resource defaults. Pilot outputs have separate paths
from production outputs. Reusing a completed or failed pilot path requires
choosing a new output name or explicitly replacing it with `--overwrite`.

Inspect jobs and their final state:

```sh
squeue -u jdehaan
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,MaxRSS
tail -n 40 next-word-prediction.JOB_ID.out
tail -n 40 next-word-prediction.JOB_ID.err
```

Before accepting a pilot, verify:

- SLURM reports `COMPLETED` and exit code `0:0`.
- The score sidecar has `status: complete`, the requested CUDA device and
  float32 precision, and matching input/output SHA-256 hashes.
- BERT scores 492 Michaelov candidates and skips eight multi-token words;
  DeepSeek, Qwen, and Llama each score all 500. Scores are finite natural log probabilities.
- CSV sentence IDs, contexts, and words match the prepared dataset.

The [results guide](../results/README.md) describes metadata and failure states.

### Validation record: October 13, 2026

SLURM job `7350085` completed successfully on `gpu3006` (NVIDIA L40S) in
2 minutes 22 seconds. It used Python 3.12.8, Torch 2.11.0+cu128, Transformers
5.19.0, and float32 inference. CUDA tensor execution, dependency checks, all
47 lightweight tests, and all five numerical integration tests passed.
The integration tests still use tiny CPU models; the pretrained pilots exercise
the CUDA pipeline.

The Michaelov pilots produced 492 BERT scores (eight expected skips) and 500
DeepSeek scores (zero skips). Automated checks verified complete metadata,
input/output hashes, counts, finite nonpositive scores, and candidate identity.
The job script and evidence are in the remote project's
`logs/validation-and-pilots.sbatch`, `logs/pilots.7350085.out`,
`logs/pilots.7350085.err`, and `logs/oscar-packages.txt`.

The earlier job `7349959` failed during CUDA initialization with the CUDA 13
wheel, before tests or scoring. Its logs are retained. The unused original
environment is at `.venv-cu130-unused`; the active `.venv` uses CUDA 12.8.

Qwen job `7350310` also completed successfully on an L40S in 2 minutes
39 seconds, including the initial checkpoint download. It produced all 500
Michaelov scores in float32 with zero skips. Its logs are
`next-word-prediction.7350310.out` and `.err`.

Copies of all three pilots, their metadata, and validation logs are in the local
`results/oscar-validation/` directory, which is ignored by Git. Local checks
verified their output hashes, candidate identities, counts, finite nonpositive
scores, and Python source hashes. The archived input's hash matches the
recorded source commit. Git normalizes the CSV's line endings, so the remote
input's byte hash differs from the Mac working copy; their parsed records are
identical.

### Llama validation: October 20, 2026

After configuring gated access, job `7662897` completed the Llama tokenizer
audit and Michaelov pilot on `gpu3101` (NVIDIA L40S) in 1 minute 12 seconds.
It used the same CUDA 12.8 environment and float32 inference as the earlier pilots,
with model and tokenizer pinned to `4e20de362430cd3b72f300e6b0f18e50e7166e08`.

The [Llama audit](llama_tokenization_audit.json) covers all 58,146 candidates:
48,197 single-token and 9,949 multi-token words, with zero boundary mismatches,
empty encodings, or unknown tokens. The pilot scored all 500 Michaelov candidates
with zero skips. Remote and local checks verified hashes, candidate identities,
finite nonpositive scores, CUDA float32 settings, and the requested revision.
Local checks also confirmed Python source hashes and equivalence between the
audited inputs and the local CSV records.

The existing `--model llama` adapter needed no changes. Its cached multi-token
scoring is now exercised with the pretrained checkpoint on CUDA. Evidence is in
`logs/llama-validation.7662897.out`, `.err`, `logs/llama-validation.sbatch`, and
`results/pilot/michaelov_2024/llama.csv` with its metadata sidecar. Copies are in
the local `results/oscar-validation/` directory.

### Pin checkpoint revisions for production

The automatic model/tokenizer revision fields were null in these pilot sidecars.
Their original metadata is preserved. The cached `main` references inspected
after the pilots match the existing tokenizer audit:

| Model    | Commit                                     |
| -------- | ------------------------------------------ |
| BERT     | `bf1420893378c390773c9452c3602fcee89f9241` |
| DeepSeek | `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562` |
| Qwen     | `d149729398750b98c0af14eb82c78cfe92750796` |
| Llama    | `4e20de362430cd3b72f300e6b0f18e50e7166e08` |

Pass the corresponding `--revision <commit>` to each production job. This pins
both model and tokenizer downloads and records the exact requested commit even
when automatic revision extraction is unavailable. Llama's audit and pilot used
the explicit commit above. The current all-model batch script cannot select
different revisions per model, so use individual CLI submissions for these
pinned runs.

## Production batch: October 20, 2026

All 16 production runs completed successfully as SLURM array `7664500`, limited
to two concurrent L40S GPUs. Each task used four CPUs, 64 GB host memory,
float32, a 20-hour time limit, and the pinned checkpoint listed above. The jobs used the
downloaded checkpoint cache in offline mode. Preflight verified the source and
input hashes, package versions, cached checkpoint files, and unused output paths.

The submitted scripts, task manifest, logs, and validation reports are under
`logs/production-20261020T160502Z/` on Oscar. Local copies of the launch files
are under `results/oscar-production/launch/`, which is ignored by Git.

| Task IDs | Dataset | Model order |
| --- | --- | --- |
| 0–3 | Michaelov | BERT, DeepSeek, Qwen, Llama |
| 4–7 | Nieuwland | BERT, DeepSeek, Qwen, Llama |
| 8–11 | Szewczyk | BERT, DeepSeek, Qwen, Llama |
| 12–15 | Peelle | BERT, DeepSeek, Qwen, Llama |

Each task validated its output after scoring. CPU job `7664501` completed
successfully and verified all 16 outputs in
`logs/production-20261020T160502Z/batch-validation.json`: **222,516 scores**
and **10,068 expected BERT exclusions**, with no failed jobs.

The scoring jobs ran from 12:12:08 to 13:25:39 on October 20, 2026 (Oscar local
time), a span of 1 hour 13 minutes 31 seconds including scheduling gaps.
The smaller datasets took 17–90 seconds per task. Peelle runtimes, including
model loading and task validation, were:

| Model | Peelle scores | Exclusions | Elapsed |
| --- | ---: | ---: | --- |
| BERT | 46,163 | 9,993 | 7m 00s |
| DeepSeek | 56,156 | 0 | 21m 14s |
| Qwen | 56,156 | 0 | 56m 26s |
| Llama | 56,156 | 0 | 14m 15s |

All 16 CSVs and sidecars have been copied into the local
`results/<dataset>/<model>.csv` locations. Independent local verification checked
SLURM completion, output and input hashes, candidate identities and multiplicity,
expected counts, finite nonpositive scores, checkpoint pins, CUDA float32
settings, evaluation mode, and Python source hashes. Remote input snapshots
match the local parsed CSV records despite line-ending differences.

The downloaded evidence, input snapshots, and local verification report are
preserved under `results/oscar-production/production-20261020T160502Z/`.
The report is `local-validation.json`; scheduler timings are in
`logs/production-20261020T160502Z/accounting.psv` within that directory.
These artifacts are ignored by Git. Original sidecars retain the Oscar paths
where the runs executed.

```sh
squeue -j 7664500,7664501
sacct -j 7664500,7664501 --format=JobID,State,ExitCode,Elapsed
cat logs/production-20261020T160502Z/batch-validation.json
```

Individual task reports are named `task-<index>.validation.json` in the same
directory. Preserve the completed outputs when running future comparisons.

## Repeating checks and runs

All four model pilots and tokenizer audits have passed. When changing
dependencies or inference code, rerun the
tests in a compute allocation using the [main README](../README.md#tests).
To repeat the Llama tokenizer audit at its validated revision:

```sh
python scripts/audit_tokenization.py --models llama \
  --revision 4e20de362430cd3b72f300e6b0f18e50e7166e08 \
  --datasets data/processed/*.csv > logs/llama-tokenization-audit-repeat.json
```

Schedule that audit through SLURM as well; it loads tokenizer files but no model
weights. Retain its report and check its exit status before Llama scoring.

Submit individual model/dataset jobs using the same launcher, pass the model's
validated `--revision`, and use a new `--output` path to preserve this batch.
Use the measured runtimes above to plan resource requests, allowing additional
time for changes in hardware, model versions, and input size.

`scripts/run_all.sh` runs all 16 combinations sequentially and must itself run
inside a GPU allocation. It stops at the first failure and does not skip existing
results. Individual jobs are easier to retry and to schedule within Oscar's
limits. Keep at most two GPU jobs running for this account's current quota.

Record the remote package versions with `python -m pip freeze` after validation.
The local CPU constraints file does not establish CUDA compatibility. Keep
float32 for the initial pilots; validate any later precision change separately.
