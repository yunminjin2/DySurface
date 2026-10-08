#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 RUN_DIR [OmegaConf overrides ...]" >&2
    exit 2
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
run_dir="$(realpath "$1")"
shift
mode="${DY_SURFACE_MODE:-validate}"
if [[ "${mode}" != validate && "${mode}" != predict ]]; then
    echo "DY_SURFACE_MODE must be validate or predict" >&2
    exit 2
fi
config="${run_dir}/config/parsed.yaml"
if [[ ! -f "${config}" ]]; then
    echo "Missing saved config: ${config}" >&2
    exit 2
fi

if [[ -n "${RESUME:-}" ]]; then
    checkpoint="$(realpath "${RESUME}")"
else
    checkpoint="$(find "${run_dir}/ckpt" -maxdepth 1 -type f -name 'epoch=*-step=*.ckpt' \
        | sed -En 's#.*step=([0-9]+)\.ckpt$#\1 &#p' \
        | sort -nr -k1,1 | head -n 1)"
    checkpoint="${checkpoint#* }"
fi
if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing checkpoint under ${run_dir}/ckpt" >&2
    exit 2
fi
step="$(basename "${checkpoint}")"
step="${step##*step=}"
step="${step%.ckpt}"

gs_dir="${RESUME_GS:-${run_dir}/output}"
if [[ ! -d "${gs_dir}/point_cloud" ]]; then
    echo "Missing GS point clouds: ${gs_dir}/point_cloud" >&2
    exit 2
fi
if [[ -n "${RESUME_GS_ITERATION:-}" ]]; then
    gs_iteration="${RESUME_GS_ITERATION}"
else
    gs_iteration="$(find "${gs_dir}/point_cloud" -mindepth 1 -maxdepth 1 -type d \
        -name 'iteration_*' -printf '%f\n' \
        | sed -En 's/^iteration_([0-9]+)$/\1/p' | sort -nr | head -n 1)"
fi
if [[ -z "${gs_iteration}" ]]; then
    echo "No positive GS iteration under ${gs_dir}/point_cloud" >&2
    exit 2
fi

gpu="${GPU:-0}"
python_bin="${PYTHON_BIN:-python}"
torch_cuda_version="$("${python_bin}" -c 'import torch; print(torch.version.cuda or "")')"
cuda_home="${DYNAMIC_SURFACE_CUDA_HOME:-${CUDA_HOME:-}}"
if [[ -z "${cuda_home}" && -d "/usr/local/cuda-${torch_cuda_version}" ]]; then
    cuda_home="/usr/local/cuda-${torch_cuda_version}"
fi
if [[ -n "${cuda_home}" ]]; then
    if [[ ! -x "${cuda_home}/bin/nvcc" ]]; then
        echo "CUDA toolkit not found at ${cuda_home}" >&2
        exit 2
    fi
    export CUDA_HOME="${cuda_home}" CUDA_PATH="${cuda_home}" CUDACXX="${cuda_home}/bin/nvcc"
    export PATH="${cuda_home}/bin:${PATH}"
    export LD_LIBRARY_PATH="${cuda_home}/lib64:${LD_LIBRARY_PATH:-}"
fi
export MAX_JOBS="${MAX_JOBS:-4}"
args=(
    --config "${config}" --gpu "${gpu}" "--${mode}"
    --resume "${checkpoint}" --resume_gs "${gs_dir}"
    --resume_iteration "${step}" --resume_gs_iteration "${gs_iteration}"
    --mesh_render
    "exp_dir=$(dirname "${run_dir}")" "trial_name=$(basename "${run_dir}")"
    "save_dir=${run_dir}/save" "ckpt_dir=${run_dir}/ckpt"
    "config_dir=${run_dir}/config" "model.using_pretrain=true"
)
if [[ -n "${DATA_ROOT:-}" ]]; then
    args+=("dataset.root_dir=$(realpath "${DATA_ROOT}")")
fi
args+=("$@")

cd "${repo_dir}"
CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" launch.py "${args[@]}"
