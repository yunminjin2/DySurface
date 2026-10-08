#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 SCENE DATA_ROOT [OmegaConf overrides ...]" >&2
    exit 2
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
scene="$1"
data_root="$(realpath "$2")"
shift 2
config="${repo_dir}/configs/dnerf/${scene}_sparse.yaml"
if [[ ! -f "${config}" ]]; then
    echo "Unknown D-NeRF scene: ${scene}" >&2
    exit 2
fi
if [[ ! -f "${data_root}/transforms_train.json" ]]; then
    echo "Missing D-NeRF transforms: ${data_root}/transforms_train.json" >&2
    exit 2
fi

gpu="${GPU:-0}"
python_bin="${PYTHON_BIN:-python}"
exp_dir="${EXP_DIR:-${repo_dir}/exp}"
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
    --config "${config}" --exp_dir "${exp_dir}" --gpu "${gpu}" --train
    "tag=${TAG:-release}" "dataset.root_dir=${data_root}"
)
if [[ -n "${GS_PRETRAIN_DIR:-}" ]]; then
    gs_pretrain_dir="$(realpath "${GS_PRETRAIN_DIR}")"
    if [[ ! -d "${gs_pretrain_dir}/point_cloud" ]]; then
        echo "Missing GS point clouds: ${gs_pretrain_dir}/point_cloud" >&2
        exit 2
    fi
    gs_iteration="$(find "${gs_pretrain_dir}/point_cloud" -mindepth 1 -maxdepth 1 -type d \
        -name 'iteration_*' -printf '%f\n' \
        | sed -En 's/^iteration_([0-9]+)$/\1/p' | sort -nr | head -n 1)"
    if [[ -z "${gs_iteration}" ]]; then
        echo "No positive GS iteration under ${gs_pretrain_dir}/point_cloud" >&2
        exit 2
    fi
    args+=(--resume_gs "${gs_pretrain_dir}" --resume_gs_iteration "${gs_iteration}" \
        "model.using_pretrain=true" "model.using_pretrain_path=${gs_pretrain_dir}")
else
    args+=("model.using_pretrain=false")
fi
args+=("$@")

cd "${repo_dir}"
CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" launch.py "${args[@]}"
