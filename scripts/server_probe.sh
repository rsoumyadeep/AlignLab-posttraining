#!/usr/bin/env bash
# ===========================================================================
# AlignLab - department server probe
# ===========================================================================
#
# PURPOSE
#   Answer, in one pass, every infrastructure question that is currently
#   UNKNOWN about the department GPU server, so that configs/env/server.yaml
#   can be filled in with VERIFIED values instead of guesses.
#
# SAFETY - this script is strictly READ-ONLY:
#   * installs nothing
#   * downloads nothing (no model, no dataset, no package)
#   * writes nothing outside a temporary directory it creates and removes
#   * modifies no user configuration, no dotfiles, no PATH
#   * requires no sudo, and never attempts privilege escalation
#   Network checks are reachability probes only (a HEAD request or a DNS
#   lookup). They transfer no project data and send no credentials.
#
# STATUS AT TIME OF COMMIT: NOT EXECUTED.
#   Written during Phase 1A on the local Windows machine. It has never been
#   run anywhere. Nothing in AlignLab may cite its output until it has
#   actually been executed and the output committed.
#
# USAGE
#   bash scripts/server_probe.sh                 # print to stdout
#   bash scripts/server_probe.sh > probe.txt     # capture for committing
#
# AFTER RUNNING
#   1. Commit the verbatim output under docs (do not edit or summarise it).
#   2. Fill configs/env/server.yaml, replacing each ??? with a verified path.
#   3. Record which probe line justified each value.
# ===========================================================================

set -u  # undefined variables are an error; deliberately NOT -e, because a
        # missing tool is information, not a reason to abort the probe.

section() {
    printf '\n========== %s ==========\n' "$1"
}

# Run a command if it exists; otherwise say so plainly. The distinction
# between "absent" and "present but failed" matters - absence of a tool is
# itself a finding.
try_cmd() {
    local label="$1"; shift
    printf -- '--- %s ---\n' "$label"
    if command -v "$1" >/dev/null 2>&1; then
        "$@" 2>&1 | head -40
    else
        printf 'NOT AVAILABLE: %s is not on PATH\n' "$1"
    fi
}

printf '===========================================================\n'
printf 'AlignLab server probe\n'
printf 'Generated: %s\n' "$(date -Is 2>/dev/null || date)"
printf 'Host: %s\n' "$(hostname 2>/dev/null || echo unknown)"
printf 'User: %s\n' "$(whoami 2>/dev/null || echo unknown)"
printf '===========================================================\n'

# ---------------------------------------------------------------------------
section "IDENTITY AND SHELL"
printf 'whoami : %s\n' "$(whoami 2>/dev/null)"
printf 'HOME   : %s\n' "${HOME:-unset}"
printf 'PWD    : %s\n' "$(pwd)"
printf 'SHELL  : %s\n' "${SHELL:-unset}"
printf 'groups : %s\n' "$(groups 2>/dev/null || echo unknown)"
printf 'id     : %s\n' "$(id 2>/dev/null || echo unknown)"

# ---------------------------------------------------------------------------
section "OS AND KERNEL"
try_cmd "uname" uname -a
try_cmd "os-release" cat /etc/os-release

# ---------------------------------------------------------------------------
section "CPU AND MEMORY"
try_cmd "nproc" nproc
try_cmd "lscpu (summary)" bash -c "lscpu | head -20"
try_cmd "free" free -h

# ---------------------------------------------------------------------------
section "GPU - CURRENT RUNTIME STATE"
# The single most important part of this probe. The May 2026 hardware report
# showed two idle A6000s; this establishes what is true NOW, including who
# else is using them.
try_cmd "nvidia-smi" nvidia-smi
printf -- '--- per-GPU query ---\n'
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,compute_cap,driver_version \
        --format=csv 2>&1
else
    printf 'NOT AVAILABLE: nvidia-smi is not on PATH\n'
fi
printf -- '--- processes currently on the GPUs (who else is here) ---\n'
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1
else
    printf 'NOT AVAILABLE\n'
fi
try_cmd "GPU topology" nvidia-smi topo -m

# ---------------------------------------------------------------------------
section "CUDA TOOLCHAIN"
try_cmd "nvcc" nvcc --version
printf -- '--- CUDA installations ---\n'
ls -d /usr/local/cuda* 2>&1 | head -10
printf -- '--- CUDA on PATH ---\n'
printf 'PATH=%s\n' "${PATH}"
printf 'LD_LIBRARY_PATH=%s\n' "${LD_LIBRARY_PATH:-unset}"

# ---------------------------------------------------------------------------
section "PYTHON"
# Determines the version floor for BOTH machines. pyproject currently declares
# requires-python >=3.10 pending this answer.
try_cmd "python3" python3 --version
try_cmd "python" python --version
try_cmd "pip3" pip3 --version
printf -- '--- other interpreters on PATH ---\n'
for v in 3.10 3.11 3.12 3.13; do
    if command -v "python$v" >/dev/null 2>&1; then
        printf 'python%s : %s\n' "$v" "$(python$v --version 2>&1)"
    fi
done
printf -- '--- venv module available? ---\n'
python3 -c "import venv; print('venv: OK')" 2>&1 | head -3

# ---------------------------------------------------------------------------
section "ENVIRONMENT MANAGERS"
try_cmd "conda" conda --version
try_cmd "mamba" mamba --version
try_cmd "uv" uv --version
try_cmd "module (environment modules)" bash -c "type module 2>&1 | head -3"
printf -- '--- module avail (if present) ---\n'
if type module >/dev/null 2>&1; then
    module avail 2>&1 | head -40
else
    printf 'NOT AVAILABLE: no environment-modules system\n'
fi

# ---------------------------------------------------------------------------
section "SCHEDULER (SLURM OR OTHER)"
# The May report contained no scheduler evidence, but it captured no software
# inventory at all - so that was absence of evidence, not evidence of absence.
# This settles it.
try_cmd "sinfo" sinfo
try_cmd "squeue (mine)" bash -c 'squeue -u "$(whoami)"'
try_cmd "sacctmgr partitions" bash -c "sacctmgr show qos format=name,maxwall -P | head -20"
try_cmd "scontrol" scontrol --version
for tool in sbatch srun salloc qsub bsub; do
    if command -v "$tool" >/dev/null 2>&1; then
        printf 'FOUND scheduler tool: %s -> %s\n' "$tool" "$(command -v $tool)"
    fi
done
printf 'SLURM_JOB_ID (set only inside a job): %s\n' "${SLURM_JOB_ID:-unset}"

# ---------------------------------------------------------------------------
section "CONTAINERS"
try_cmd "apptainer" apptainer --version
try_cmd "singularity" singularity --version
try_cmd "podman" podman --version
printf -- '--- docker (daemon runs as root; membership is what matters) ---\n'
if command -v docker >/dev/null 2>&1; then
    printf 'docker binary: %s\n' "$(command -v docker)"
    if groups 2>/dev/null | tr ' ' '\n' | grep -qx docker; then
        printf 'docker group membership: YES\n'
    else
        printf 'docker group membership: NO (docker would need sudo)\n'
    fi
else
    printf 'NOT AVAILABLE: docker is not on PATH\n'
fi

# ---------------------------------------------------------------------------
section "STORAGE AND QUOTA"
# The May report showed /data at 74% full and SHARED. Checkpoints and the HF
# cache must go on a large volume - never on the root partition.
try_cmd "df" df -h
printf -- '--- home directory usage ---\n'
du -sh "${HOME}" 2>&1 | head -3
printf -- '--- quota (if enforced) ---\n'
if command -v quota >/dev/null 2>&1; then
    quota -s 2>&1 | head -10
else
    printf 'NOT AVAILABLE: quota command absent (does not prove no quota)\n'
fi
printf -- '--- writability of candidate roots ---\n'
for candidate in "${HOME}" /data /scratch /tmp; do
    if [ -d "$candidate" ]; then
        if [ -w "$candidate" ]; then
            printf '%-24s EXISTS, WRITABLE\n' "$candidate"
        else
            printf '%-24s EXISTS, NOT WRITABLE\n' "$candidate"
        fi
    else
        printf '%-24s DOES NOT EXIST\n' "$candidate"
    fi
done

# ---------------------------------------------------------------------------
section "NETWORK REACHABILITY"
# Decides whether packages and the base model can be fetched on the server at
# all, or whether an offline transfer strategy is needed. Reachability only -
# nothing is uploaded.
probe_url() {
    local url="$1"
    if command -v curl >/dev/null 2>&1; then
        local code
        code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 12 -I "$url" 2>/dev/null)
        if [ "$code" != "000" ] && [ -n "$code" ]; then
            printf '%-32s REACHABLE (HTTP %s)\n' "$url" "$code"
        else
            printf '%-32s NOT REACHABLE\n' "$url"
        fi
    else
        printf '%-32s UNKNOWN (curl absent)\n' "$url"
    fi
}
probe_url "https://pypi.org/simple/"
probe_url "https://huggingface.co"
probe_url "https://api.wandb.ai"
probe_url "https://github.com"
printf -- '--- proxy variables ---\n'
for var in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy; do
    printf '%-14s = %s\n' "$var" "$(eval echo \${$var:-unset})"
done
try_cmd "git" git --version

# ---------------------------------------------------------------------------
section "LONG-RUN TOOLING"
try_cmd "tmux" tmux -V
try_cmd "screen" screen --version
printf -- '--- signals available to Python (SIGUSR1 is the preemption signal) ---\n'
python3 -c "import signal; print(sorted(s.name for s in signal.valid_signals() if hasattr(s,'name')))" 2>&1 | head -5

# ---------------------------------------------------------------------------
section "EXISTING PYTHON ML STACK (system-wide)"
# Informational only. AlignLab will use its own isolated environment; this just
# shows what is already present.
python3 - <<'PYEOF' 2>&1 | head -30
import importlib.util
mods = ["torch", "transformers", "datasets", "peft", "trl", "accelerate",
        "bitsandbytes", "wandb", "hydra", "omegaconf", "numpy", "flash_attn"]
for m in mods:
    try:
        spec = importlib.util.find_spec(m)
    except Exception:
        spec = None
    if spec is None:
        print(f"{m}: NOT INSTALLED")
        continue
    try:
        mod = __import__(m)
        print(f"{m}: {getattr(mod, '__version__', 'unknown version')}")
    except Exception as exc:
        print(f"{m}: import FAILED ({type(exc).__name__})")
PYEOF

# ---------------------------------------------------------------------------
section "TORCH CUDA CHECK (only if torch is already present)"
python3 - <<'PYEOF' 2>&1 | head -20
try:
    import torch
except Exception as exc:
    print(f"torch not importable: {type(exc).__name__}: {exc}")
else:
    print("torch:", torch.__version__)
    print("torch.version.cuda:", torch.version.cuda)
    print("cuda.is_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"  gpu{i}: {p.name} {p.total_memory/1024**3:.1f} GiB cc={p.major}.{p.minor}")
        try:
            print("bf16 supported:", torch.cuda.is_bf16_supported())
        except Exception:
            print("bf16 supported: could not determine")
PYEOF

# ---------------------------------------------------------------------------
section "PROBE COMPLETE"
printf 'Every line above is an OBSERVATION from this machine at this moment.\n'
printf 'Commit this output verbatim. Do not summarise it, and do not treat\n'
printf 'any value as permanent - a shared server changes.\n'
