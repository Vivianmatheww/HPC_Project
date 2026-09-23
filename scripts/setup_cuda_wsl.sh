#!/usr/bin/env bash
# Installs the CUDA toolkit on WSL2 Ubuntu 22.04. Run once: bash scripts/setup_cuda_wsl.sh
# The GPU driver comes from Windows. Never install a Linux driver (`cuda`, `cuda-drivers`,
# `nvidia-driver-*`) inside WSL, because it would break GPU access.
set -euo pipefail

CUDA_VER=13-3            # newest toolkit in NVIDIA's wsl-ubuntu repo; driver 615.78 supports up to 13.4
CUDA_DIR=/usr/local/cuda-${CUDA_VER/-/.}

echo "==> Adding NVIDIA wsl-ubuntu apt repository"
tmp=$(mktemp -d)
wget -q -O "$tmp/cuda-keyring.deb" \
  https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i "$tmp/cuda-keyring.deb"
sudo apt-get update

echo "==> Installing cuda-toolkit-${CUDA_VER} (toolkit only, no driver) and build tools"
sudo apt-get install -y "cuda-toolkit-${CUDA_VER}" build-essential git

echo "==> Adding CUDA to PATH in ~/.bashrc"
if ! grep -q "# >>> cuda >>>" ~/.bashrc; then
  cat >> ~/.bashrc <<EOF
# >>> cuda >>>
export PATH=${CUDA_DIR}/bin:\$PATH
export LD_LIBRARY_PATH=${CUDA_DIR}/lib64:/usr/lib/wsl/lib:\${LD_LIBRARY_PATH:-}
# <<< cuda <<<
EOF
fi
export PATH=${CUDA_DIR}/bin:$PATH

echo "==> Checking"
nvcc --version | tail -2
ls ${CUDA_DIR}/lib64/libcusparse.so >/dev/null && echo "cuSPARSE found"
command -v nsys >/dev/null && echo "Nsight Systems: $(nsys --version)" || echo "Nsight Systems not on PATH"
command -v ncu  >/dev/null && echo "Nsight Compute: $(ncu --version | tail -1)" || echo "Nsight Compute not on PATH"
echo "Done. Open a new terminal (or 'source ~/.bashrc') to pick up PATH."
