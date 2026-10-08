#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DY_SURFACE_MODE=predict exec "${repo_dir}/val.sh" "$@"
