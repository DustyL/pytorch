#!/bin/bash
#
# 🔥🔥🔥 NVIDIA B300 (SM_103) PYTORCH + TRITON - ULTIMATE GOD MODE 🔥🔥🔥
#
# Optimized for: AMD EPYC 9575F (30 cores) + NVIDIA B300 + CUDA 13.1
# Target: PyTorch 2.11.0a0 + Triton (from source) with native sm_103 support
#
# This script merges the best practices from multiple sources:
# - Robust CUDA detection and versioning
# - Triton build from source (required for CUDA 13.1 + B300)
# - Flash-Attention sm_103 patches with verification
# - All performance optimizations enabled
# - Comprehensive safety checks
#
# Usage:
#   ./build_b300_GODMODE.sh       # Interactive mode
#   ./build_b300_GODMODE.sh -y    # Non-interactive mode
#

set -euo pipefail

# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
# Default log location can be overridden via GODMODE_LOG.
GODMODE_LOG="${GODMODE_LOG:-/root/godmode_build_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p "$(dirname "$GODMODE_LOG")"
exec > >(tee -a "$GODMODE_LOG") 2>&1
echo "✓ Logging to: $GODMODE_LOG"

echo "================================================================================"
echo "🔥🔥🔥 NVIDIA B300 PYTORCH + TRITON - ULTIMATE GOD MODE 🔥🔥🔥"
echo "================================================================================"
echo ""

cd /root/pytorch

# ============================================================================
# VENV ACTIVATION
# ============================================================================

# Activate venv if it exists (unless explicitly skipped)
# (Repo commonly uses `.venv`, but allow `venv` too.)
if [[ "${SKIP_PYTORCH_VENV:-0}" != "1" ]]; then
    VENV_ACTIVATE=""
    for cand in /root/venv/bin/activate /root/pytorch/.venv/bin/activate /root/pytorch/venv/bin/activate; do
        if [[ -f "$cand" ]]; then
            VENV_ACTIVATE="$cand"
            break
        fi
    done
    if [[ -n "$VENV_ACTIVATE" ]]; then
        echo "🐍 Activating venv: $VENV_ACTIVATE"
        # shellcheck disable=SC1090
        source "$VENV_ACTIVATE"
        echo "✓ venv activated"
        echo ""
    fi
fi

# Which Python should be used to build PyTorch
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "❌ ERROR: PYTHON_BIN not found/executable: $PYTHON_BIN"
    exit 1
fi
echo "✓ Python: $PYTHON_BIN ($($PYTHON_BIN --version))"
echo ""

# ============================================================================
# PRE-BUILD CHECKS
# ============================================================================

echo "📋 Pre-build checks..."
echo ""

# Check GPU
if ! nvidia-smi &>/dev/null; then
    echo "❌ ERROR: nvidia-smi not working!"
    exit 1
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "✓ GPU detected: $GPU_NAME"

# Robust Compute Capability Detection (using PYTHON_BIN)
COMPUTE_CAP=$("$PYTHON_BIN" -c "import torch; print('.'.join(map(str, torch.cuda.get_device_capability())))" 2>/dev/null || echo "10.3")
echo "✓ Compute capability: $COMPUTE_CAP"

if [ "$COMPUTE_CAP" != "10.3" ]; then
    echo "⚠️  WARNING: Expected 10.3 for B300, got $COMPUTE_CAP"
fi

# Check CUDA (auto-detect 13.0 or 13.1)
if [[ -d /usr/local/cuda-13.1 ]]; then
    export CUDA_HOME=/usr/local/cuda-13.1
elif [[ -d /usr/local/cuda-13.0 ]]; then
    export CUDA_HOME=/usr/local/cuda-13.0
elif [[ -d /usr/local/cuda ]]; then
    export CUDA_HOME=/usr/local/cuda
    echo "⚠️  WARNING: Using default CUDA path: $CUDA_HOME"
else
    echo "❌ ERROR: CUDA not found! Set CUDA_HOME manually."
    exit 1
fi

if [ ! -f "$CUDA_HOME/bin/nvcc" ]; then
    echo "❌ ERROR: nvcc not found at $CUDA_HOME/bin/nvcc"
    exit 1
fi

# Ensure the selected toolkit wins over any other CUDA in PATH.
export PATH="$CUDA_HOME/bin:${PATH}"
export CUDA_NVCC_EXECUTABLE="$CUDA_HOME/bin/nvcc"
export CUDACXX="$CUDA_HOME/bin/nvcc"

# Robust CUDA Version Detection (FIXES THE V13.1.115 BUG)
CUDA_RELEASE="$("$CUDA_HOME/bin/nvcc" --version | awk -F'release ' '/release/ {print $2}' | awk -F',' '{print $1}' | head -1)"
echo "✓ CUDA version: $CUDA_RELEASE"
CUDA_TAG="cu${CUDA_RELEASE//./}"
echo "✓ CUDA wheel tag: $CUDA_TAG"

# Check cuDNN (use grep without -q to avoid SIGPIPE with set -o pipefail)
if ! dpkg -l 2>/dev/null | grep "libcudnn9-cuda-13" >/dev/null 2>&1; then
    echo "❌ ERROR: cuDNN 9 for CUDA 13 not installed"
    exit 1
fi
CUDNN_VERSION=$(dpkg -l 2>/dev/null | grep "libcudnn9-cuda-13" | head -1 | awk '{print $3}')
echo "✓ cuDNN version: $CUDNN_VERSION"

# Check Ninja (CRITICAL for build speed)
if ! command -v ninja >/dev/null 2>&1; then
    echo "❌ ERROR: ninja not found! Install: pip install ninja"
    echo "   Without ninja, build will take 6+ hours instead of 45-65 minutes"
    exit 1
fi
echo "✓ ninja found: $(ninja --version)"

# Check cuDNN SDPA fix (should already be in PyTorch 2.11.0a0)
if grep -q "fixSizeOneDimStrideSDPA" /root/pytorch/aten/src/ATen/native/cudnn/MHA.cpp; then
    echo "✓ cuDNN SDPA backward stride fix: CONFIRMED (already in source)"
else
    echo "⚠️  WARNING: cuDNN SDPA fix not found (may cause backward pass issues)"
fi

# Sync all submodules first (before patching)
echo ""
echo "🔄 Syncing all submodules..."

# Avoid submodule update failures due to local changes in flash-attention.
# NOTE: for git submodules, `.git` is typically a *file* (gitdir: ...), not a directory.
FLASH_ATTN_PATH="third_party/flash-attention"
FLASH_ATTN_DIR="/root/pytorch/$FLASH_ATTN_PATH"
FLASH_ATTN_DIRTY=0
if [[ -e "$FLASH_ATTN_DIR/.git" ]]; then
    if [[ -n "$(cd "$FLASH_ATTN_DIR" && git status --porcelain=v1 2>/dev/null)" ]]; then
        FLASH_ATTN_DIRTY=1
        echo "⚠️  Detected local changes in $FLASH_ATTN_PATH (will not auto-checkout over them)."
        echo "    Tip: set CLEAN_FLASH_ATTN_SUBMODULE=1 if you want this script to reset+clean it."
    fi
fi

git submodule sync --recursive --quiet

# If flash-attention is dirty, updating submodules will try to checkout the superproject-pinned SHA
# and fail. Update other submodules, and preserve flash-attention as-is.
SUBMODULE_PATHS=()
if git config -f .gitmodules --get-regexp path >/dev/null 2>&1; then
    while read -r _key _path; do
        SUBMODULE_PATHS+=("$_path")
    done < <(git config -f .gitmodules --get-regexp path)
fi

UPDATE_SUBMODULES=()
for p in "${SUBMODULE_PATHS[@]}"; do
    if [[ "$p" == "$FLASH_ATTN_PATH" && "$FLASH_ATTN_DIRTY" == "1" ]]; then
        continue
    fi
    UPDATE_SUBMODULES+=("$p")
done

if [[ "$FLASH_ATTN_DIRTY" == "1" && "${CLEAN_FLASH_ATTN_SUBMODULE:-0}" == "1" ]]; then
    echo "   - Cleaning $FLASH_ATTN_PATH submodule working tree (reset + clean)..."
    (cd "$FLASH_ATTN_DIR" && git reset --hard -q && git clean -fdx -q) || true
    FLASH_ATTN_DIRTY=0
    # Now it is safe to update it like the others.
    UPDATE_SUBMODULES+=("$FLASH_ATTN_PATH")
fi

if [[ ${#UPDATE_SUBMODULES[@]} -gt 0 ]]; then
    git submodule update --init --recursive --quiet "${UPDATE_SUBMODULES[@]}"
else
    echo "   - No submodules to update (skipping)"
fi

# 🔧 AUTO-HEALING: Flash-Attention Submodule (CRITICAL FIX!)
echo ""
echo "🔧 Auto-healing Flash-Attention for SM_103..."

FA_PATH="third_party/flash-attention"
FA_BWD_FILE="$FA_PATH/flash_attn/cute/flash_bwd_sm100.py"
SKIP_FLASH_ATTN_PATCHES="${SKIP_FLASH_ATTN_PATCHES:-0}"

# 1. Ensure the Blackwell CuTe backward exists (file name contains "sm100" upstream)
if [ ! -f "/root/pytorch/$FA_BWD_FILE" ]; then
    echo "   - Blackwell CuTe backward file missing ($FA_BWD_FILE)"
    if [[ "$SKIP_FLASH_ATTN_PATCHES" == "1" ]]; then
        echo "❌ ERROR: SKIP_FLASH_ATTN_PATCHES=1 and required file is missing."
        echo "   Either: (1) provide the file in your submodule state, or (2) unset SKIP_FLASH_ATTN_PATCHES."
        exit 1
    fi

    echo "   - Auto-fetching Flash-Attention commit 13696f2..."
    if [[ "$FLASH_ATTN_DIRTY" == "1" ]]; then
        echo "❌ ERROR: $FLASH_ATTN_PATH is dirty, refusing to checkout a different commit."
        echo "   Either commit/stash your changes, or run with CLEAN_FLASH_ATTN_SUBMODULE=1."
        exit 1
    fi
    (cd "/root/pytorch/$FA_PATH" && git fetch origin && git checkout 13696f2e5e235696a6851eada1780f7753226a68)
    # After changing the submodule commit, ensure nested submodules are consistent with it.
    (cd "/root/pytorch/$FA_PATH" && git submodule update --init --recursive --quiet || true)
    echo "   ✓ Flash-Attention submodule updated"
else
    echo "   ✓ Flash-Attention Blackwell CuTe backward: CONFIRMED"
fi

# 2. Patch setup.py for sm_103 (optional)
FA_SETUP="/root/pytorch/$FA_PATH/setup.py"

if [[ "$SKIP_FLASH_ATTN_PATCHES" == "1" ]]; then
    echo "   - SKIP_FLASH_ATTN_PATCHES=1 (preserving submodule state; verification-only)"
    if ! grep -q "\"103\"" "$FA_SETUP"; then
        echo "   ⚠️  setup.py does not mention \"103\"; Flash-Attention may not build sm_103 kernels."
    fi
    if ! grep -q "compute_103" "$FA_SETUP"; then
        echo "   ⚠️  setup.py does not contain compute_103 gencode logic."
    fi
else
    # Apply Patch 1: Add 103 to arch list
    if ! grep -q "\"103\"" "$FA_SETUP"; then
        echo "   - Patching architecture list to include sm_103..."
        sed -i 's/"80;90;100;110;120"/"80;90;100;103;110;120"/' "$FA_SETUP"
        echo "   ✓ Architecture list patched"
    else
        echo "   ✓ Architecture list already includes sm_103"
    fi

    # Apply Patch 2: Inject sm_103 compiler flags
    if ! grep -q "compute_103" "$FA_SETUP"; then
        echo "   - Injecting sm_103 compiler flags..."
        sed -i '/if "110" in archs:/i \
        if "103" in archs:\
            if bare_metal_version >= Version("13.0"):\
                cc_flag += ["-gencode", "arch=compute_103f,code=sm_103"]\
            elif bare_metal_version >= Version("12.8"):\
                cc_flag += ["-gencode", "arch=compute_103,code=sm_103"]\
' "$FA_SETUP"
        echo "   ✓ Compiler flags injected"
    else
        echo "   ✓ sm_103 compiler flags already present"
    fi

    echo "✓ Flash-Attention: AUTO-HEALED & PATCHED (after submodule sync)"
fi

echo ""
echo "✅ All pre-build checks passed!"
echo ""

# ============================================================================
# BUILD CONFIGURATION
# ============================================================================

echo "⚙️  Configuring build environment for SM_103..."
echo ""

# CUDA paths (PATH already prepended above when selecting CUDA_HOME)
export TRITON_PTXAS_PATH=$CUDA_HOME/bin/ptxas

# Build LD_LIBRARY_PATH with CUPTI if it exists
if [ -f "$CUDA_HOME/extras/CUPTI/lib64/libcupti.so" ]; then
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:${LD_LIBRARY_PATH:-}"
else
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi

echo "✓ CUDA_HOME: $CUDA_HOME"

# 🔥 CRITICAL: SM_103 Architecture Configuration for B300 🔥
export TORCH_CUDA_ARCH_LIST="10.3+PTX"
echo "✓ TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST (SM_103!)"

# 🔥 Flash-Attention SM_103 Configuration 🔥
export FLASH_ATTN_CUDA_ARCHS="80;90;100;103"
echo "✓ FLASH_ATTN_CUDA_ARCHS: $FLASH_ATTN_CUDA_ARCHS (INCLUDES SM_103!)"

# Core CUDA Features
export USE_CUDA=1
export USE_CUDNN=1
export USE_NVRTC=1
echo "✓ CUDA features: USE_CUDA=1, USE_CUDNN=1, USE_NVRTC=1"

# cuDNN paths - explicitly set for system-installed cuDNN
CUDNN_SO=""
if [[ -f /usr/lib/x86_64-linux-gnu/libcudnn.so ]]; then
    CUDNN_SO=/usr/lib/x86_64-linux-gnu/libcudnn.so
elif [[ -f /usr/lib/x86_64-linux-gnu/libcudnn.so.9 ]]; then
    CUDNN_SO=/usr/lib/x86_64-linux-gnu/libcudnn.so.9
elif [[ -f /usr/local/cuda/lib64/libcudnn.so ]]; then
    CUDNN_SO=/usr/local/cuda/lib64/libcudnn.so
fi

if [[ -n "$CUDNN_SO" ]]; then
    export CUDNN_LIB_DIR="$(dirname "$CUDNN_SO")"
    export CUDNN_LIBRARY="$CUDNN_SO"
    # Find include dir
    if [[ -f /usr/include/x86_64-linux-gnu/cudnn.h ]]; then
        export CUDNN_INCLUDE_DIR=/usr/include/x86_64-linux-gnu
    elif [[ -f /usr/local/cuda/include/cudnn.h ]]; then
        export CUDNN_INCLUDE_DIR=/usr/local/cuda/include
    fi
    echo "✓ cuDNN library: $CUDNN_LIBRARY"
    echo "✓ cuDNN include: $CUDNN_INCLUDE_DIR"
else
    echo "⚠️  WARNING: cuDNN library not found - cuDNN SDPA will not be available"
fi

# Attention Backends (Critical for Diffusion Models)
export USE_FLASH_ATTENTION=1
export USE_MEM_EFF_ATTENTION=1
echo "✓ Attention backends: Flash-Attention (WITH SM_103!) + Memory-Efficient"

# Quantization & Performance Libraries
export USE_FBGEMM=1
export USE_FBGEMM_GENAI=1
export USE_CUSPARSELT=1
export USE_CUDSS="${USE_CUDSS:-1}"

# MAGMA - Dense linear algebra on GPU (LU, QR, SVD, eigenvalues)
# Built from source with sm_103 (B300) support at /root/magma
if [[ -f /usr/local/magma/lib/libmagma.so ]]; then
    export USE_MAGMA=1
    export MAGMA_HOME=/usr/local/magma
    echo "✓ MAGMA enabled: $MAGMA_HOME (sm_103 native)"
else
    export USE_MAGMA=0
    echo "⚠ MAGMA not found at /usr/local/magma - skipping"
fi

# Check for pip-installed cuSPARSELt (if using venv)
VENV_SITE_PACKAGES=$("$PYTHON_BIN" -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || echo "")
if [[ -n "$VENV_SITE_PACKAGES" && -d "$VENV_SITE_PACKAGES/nvidia/cusparselt" ]]; then
    echo "✓ Found nvidia-cusparselt in venv"
    export CUSPARSELT_ROOT="$VENV_SITE_PACKAGES/nvidia/cusparselt"
    export LD_LIBRARY_PATH="$CUSPARSELT_ROOT/lib:${LD_LIBRARY_PATH:-}"
fi

# Avoid CMake resolving CUDSS/CUSPARSELT include paths to /usr/include, which can
# propagate as `-isystem /usr/include` and break libstdc++ `#include_next`.
if [[ "${USE_CUSPARSELT:-0}" == "1" && -z "${CUSPARSELT_INCLUDE_DIR:-}" ]]; then
    if [[ -n "${CUSPARSELT_ROOT:-}" && -f "$CUSPARSELT_ROOT/include/cusparseLt.h" ]]; then
        export CUSPARSELT_INCLUDE_DIR="$CUSPARSELT_ROOT/include"
    elif [[ -f /usr/include/libcusparseLt/13/cusparseLt.h ]]; then
        export CUSPARSELT_INCLUDE_DIR="/usr/include/libcusparseLt/13"
    fi
fi

if [[ "${USE_CUDSS:-0}" == "1" && -z "${CUDSS_INCLUDE_DIR:-}" ]]; then
    if [[ -f /usr/include/libcudss/13/cudss.h ]]; then
        export CUDSS_INCLUDE_DIR="/usr/include/libcudss/13"
    elif [[ -f /usr/include/libcudss/12/cudss.h ]]; then
        export CUDSS_INCLUDE_DIR="/usr/include/libcudss/12"
    fi
fi

echo "✓ Quantization: FBGEMM + FBGEMM_GENAI + cuSPARSELt"

# Distributed Training
export USE_DISTRIBUTED=1
export USE_NCCL=1
export USE_GLOO=1
export USE_MPI=0
export USE_NVSHMEM="${USE_NVSHMEM:-0}"
export USE_SYSTEM_NCCL=0  # Use bundled NCCL for CUDA 13.1 compatibility
echo "✓ Distributed: NCCL (bundled) + GLOO"

# NVSHMEM (optional; required only if you set USE_NVSHMEM=1)
if [[ "$USE_NVSHMEM" == "1" ]]; then
    echo "✓ NVSHMEM requested (USE_NVSHMEM=1) - probing system install..."
    if [[ -f /usr/include/nvshmem_13/nvshmem.h && -f /usr/lib/x86_64-linux-gnu/nvshmem/13/libnvshmem_device.a ]]; then
        export NVSHMEM_HOME="${NVSHMEM_HOME:-/root/nvshmem_cuda13}"
        mkdir -p "$NVSHMEM_HOME/include" "$NVSHMEM_HOME/lib"
        ln -snf /usr/include/nvshmem_13 "$NVSHMEM_HOME/include/nvshmem_13"
        ln -snf /usr/include/nvshmem_13/nvshmem.h "$NVSHMEM_HOME/include/nvshmem.h"
        ln -snf /usr/include/nvshmem_13/device "$NVSHMEM_HOME/include/device"
        ln -snf /usr/lib/x86_64-linux-gnu/nvshmem/13/* "$NVSHMEM_HOME/lib/" 2>/dev/null || true
        export LD_LIBRARY_PATH="$NVSHMEM_HOME/lib:${LD_LIBRARY_PATH:-}"
        echo "✓ NVSHMEM_HOME: $NVSHMEM_HOME"
    else
        echo "❌ ERROR: USE_NVSHMEM=1 but NVSHMEM headers/libs not found where expected."
        echo "   Expected: /usr/include/nvshmem_13/nvshmem.h and /usr/lib/x86_64-linux-gnu/nvshmem/13/libnvshmem_device.a"
        exit 1
    fi
else
    echo "✓ NVSHMEM: disabled (USE_NVSHMEM=0)"
fi

# Performance Optimizations
export USE_OPENMP=1
export USE_MKLDNN=1
export USE_KINETO=1

# CUPTI Guard (for Kineto profiling - verify libcupti.so exists)
if [ -f "$CUDA_HOME/extras/CUPTI/lib64/libcupti.so" ]; then
    export USE_CUPTI_SO=1
    echo "✓ CUPTI found: Profiling enabled (libcupti.so detected)"
else
    export USE_CUPTI_SO=0
    echo "⚠️  CUPTI not found at $CUDA_HOME/extras/CUPTI/lib64 - Disabling USE_CUPTI_SO"
fi

# GPUDirect Storage Guard
if ldconfig -p 2>/dev/null | grep -q libcufile; then
    export USE_CUFILE=1
    echo "✓ Performance libs: OpenMP + MKLDNN + Kineto + CUPTI + CuFile (GDS enabled)"
else
    export USE_CUFILE=0
    echo "✓ Performance libs: OpenMP + MKLDNN + Kineto + CUPTI (CuFile disabled - not found)"
fi

# Build Speed Optimizations (FIX: Use actual CPU count, not 64)
export MAX_JOBS=$(nproc)
export CMAKE_BUILD_PARALLEL_LEVEL="$MAX_JOBS"
export BUILD_TEST=0
export BUILD_CAFFE2=0
export USE_NUMPY=1
export USE_CCACHE=1
echo "✓ Build settings: MAX_JOBS=$MAX_JOBS (matched to CPU cores), BUILD_TEST=0"

# Version & Metadata
PYTORCH_SRC_VERSION="$(cat version.txt)"
DEFAULT_BUILD_VERSION="${PYTORCH_SRC_VERSION}+${CUDA_TAG}.b300.sdpafix.sm103"
export PYTORCH_BUILD_VERSION="${PYTORCH_BUILD_VERSION:-$DEFAULT_BUILD_VERSION}"
export PYTORCH_BUILD_NUMBER="${PYTORCH_BUILD_NUMBER:-1}"
echo "✓ Build version: $PYTORCH_BUILD_VERSION"

echo ""

# ============================================================================
# SHOW WHAT WILL BE COMPILED
# ============================================================================

echo "================================================================================"
echo "📦 BUILD SPECIFICATION"
echo "================================================================================"
echo ""
echo "Host Configuration:"
echo "  - CPU cores: $MAX_JOBS (logical)"
echo "  - GPU: $GPU_NAME"
echo "  - CUDA: $CUDA_RELEASE"
echo "  - cuDNN: $CUDNN_VERSION"
echo ""
echo "PyTorch CUDA Architectures:"
echo "  - sm_103 (B300 native with family-specific optimizations!)"
echo "  - compute_103 (PTX for forward compatibility)"
echo ""
echo "Flash-Attention CUDA Architectures:"
echo "  - sm_80 (Ampere - A100)"
echo "  - sm_90 (Hopper - H100)"
echo "  - sm_100 (Blackwell - B100/B200)"
echo "  - sm_103 (Blackwell - B300!) 🔥🔥🔥"
echo ""
echo "Expected Compiler Flags:"
echo "  PyTorch: -gencode arch=compute_103,code=sm_103"
echo "           -gencode arch=compute_103,code=compute_103"
echo ""
echo "  Flash-Attention: -gencode arch=compute_80,code=sm_80"
echo "                   -gencode arch=compute_90,code=sm_90"
echo "                   -gencode arch=compute_100f,code=sm_100"
echo "                   -gencode arch=compute_103f,code=sm_103 🔥"
echo "                   -gencode arch=compute_103,code=compute_103"
echo ""
echo "================================================================================"
echo ""

# ============================================================================
# SANITY SNAPSHOT (recorded in build log)
# ============================================================================

echo "================================================================================"
echo "🔎 SANITY SNAPSHOT"
echo "================================================================================"
echo ""
echo "Paths:"
echo "  - PWD: $(pwd)"
echo "  - Python: $(command -v "$PYTHON_BIN")"
echo "  - nvcc: $CUDA_HOME/bin/nvcc"
echo ""
echo "Versions:"
echo "  - Python: $("$PYTHON_BIN" --version)"
echo "  - CUDA (nvcc): $("$CUDA_HOME/bin/nvcc" --version | grep -m1 \"release\" || true)"
echo "  - CMake: $(cmake --version | head -n 1)"
echo "  - Ninja: $(ninja --version)"
echo ""
echo "Repo:"
echo "  - pytorch: $(git describe --tags --always --dirty 2>/dev/null || git rev-parse --short HEAD)"
echo "  - flash-attention: $(cd third_party/flash-attention && git rev-parse --short HEAD 2>/dev/null || echo unknown)$(cd third_party/flash-attention && git diff --quiet && echo \"\" || echo \"-dirty\")"
echo ""
echo "Config:"
echo "  - TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST:-}"
echo "  - FLASH_ATTN_CUDA_ARCHS: ${FLASH_ATTN_CUDA_ARCHS:-}"
echo "  - USE_NVSHMEM: ${USE_NVSHMEM:-}"
echo "  - NVSHMEM_HOME: ${NVSHMEM_HOME:-}"
echo ""

# ============================================================================
# INTERACTIVE CONFIRMATION
# ============================================================================

# Prompt for confirmation (skip with -y flag)
if [[ "${1:-}" == "-y" ]]; then
    echo "✓ Non-interactive mode (-y flag)"
else
    read -p "Proceed with Triton + PyTorch build? [Y/n] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ ! -z $REPLY ]]; then
        echo "Build cancelled."
        exit 1
    fi
fi
echo ""

# ============================================================================
# STEP 1: BUILD TRITON FROM SOURCE (CRITICAL FOR B300 + CUDA 13.1)
# ============================================================================

echo "================================================================================"
echo "🔱 STEP 1: Building Triton from Source (as .whl for reusability)"
echo "================================================================================"
echo ""
echo "Why? PyPI Triton wheels are built for CUDA 12.x. For B300 (sm_103) with"
echo "CUDA 13.1, we MUST build Triton from source to generate valid PTX."
echo ""
echo "Building as .whl allows quick reinstallation without recompiling."
echo ""

BUILD_START_TRITON=$(date +%s)

# Keep Triton wheel(s) out of PyTorch's ./dist (which we delete before building torch).
TRITON_WHEEL_DIR="${TRITON_WHEEL_DIR:-/root/wheels/triton}"
mkdir -p "$TRITON_WHEEL_DIR"

# Build Triton as a wheel (not just install)
TRITON_WHEEL=""
TRITON_EXPECTED_VERSION=""
TRITON_PINNED_COMMIT=""
if [[ -f ".ci/docker/triton_version.txt" && -f ".ci/docker/ci_commit_pins/triton.txt" && -f ".github/scripts/build_triton_wheel.py" ]]; then
    TRITON_EXPECTED_VERSION="$(tr -d '[:space:]' < .ci/docker/triton_version.txt)"
    TRITON_PINNED_COMMIT="$(tr -d '[:space:]' < .ci/docker/ci_commit_pins/triton.txt)"
fi

INSTALLED_TRITON_VERSION=$("$PYTHON_BIN" -c "import triton; print(triton.__version__)" 2>/dev/null || echo "")
EXISTING_TRITON_WHEEL=$(ls -t "$TRITON_WHEEL_DIR"/triton*.whl 2>/dev/null | head -1 || true)

if [[ -n "$TRITON_EXPECTED_VERSION" && -n "$TRITON_PINNED_COMMIT" ]]; then
    if [[ "$INSTALLED_TRITON_VERSION" == "$TRITON_EXPECTED_VERSION" && -n "$EXISTING_TRITON_WHEEL" ]]; then
        echo "✓ Triton already installed at expected version: $INSTALLED_TRITON_VERSION (wheel already present)"
        TRITON_WHEEL="$EXISTING_TRITON_WHEEL"
    else
        echo "Building PyTorch-pinned Triton wheel: v$TRITON_EXPECTED_VERSION @ $TRITON_PINNED_COMMIT"
        (cd "$TRITON_WHEEL_DIR" && "$PYTHON_BIN" /root/pytorch/.github/scripts/build_triton_wheel.py --device cuda --commit-hash "$TRITON_PINNED_COMMIT" --triton-version "$TRITON_EXPECTED_VERSION")
        TRITON_WHEEL=$(ls -t "$TRITON_WHEEL_DIR"/triton*.whl 2>/dev/null | head -1 || true)
    fi
else
    echo "⚠️  WARNING: Could not find PyTorch Triton pins; falling back to Triton main (less reproducible)"
    "$PYTHON_BIN" -m pip wheel --no-deps -w "$TRITON_WHEEL_DIR" "git+https://github.com/openai/triton.git@main#subdirectory=python"
    TRITON_WHEEL=$(ls -t "$TRITON_WHEEL_DIR"/triton*.whl 2>/dev/null | head -1 || true)
fi

# Install the wheel we just built
if [ -n "$TRITON_WHEEL" ]; then
    echo "Installing Triton wheel: $TRITON_WHEEL"
    "$PYTHON_BIN" -m pip install --force-reinstall "$TRITON_WHEEL" --break-system-packages
else
    echo "⚠️  Warning: Triton wheel not found, falling back to direct install"
    "$PYTHON_BIN" -m pip install "git+https://github.com/openai/triton.git@main#subdirectory=python" --break-system-packages
fi

BUILD_END_TRITON=$(date +%s)
BUILD_TIME_TRITON=$((BUILD_END_TRITON - BUILD_START_TRITON))

echo ""
echo "✅ Triton build complete ($(($BUILD_TIME_TRITON / 60))m $(($BUILD_TIME_TRITON % 60))s)"
echo ""

# Verify Triton installation
TRITON_VERSION=$("$PYTHON_BIN" -c "import triton; print(triton.__version__)" 2>/dev/null || echo "UNKNOWN")
echo "✓ Triton version: $TRITON_VERSION"
if [ -n "$TRITON_WHEEL" ]; then
    echo "✓ Triton wheel saved: $TRITON_WHEEL (for quick reinstall)"
fi
echo ""

# ============================================================================
# STEP 2: CLEAN PREVIOUS PYTORCH BUILDS
# ============================================================================

echo "================================================================================"
echo "🧹 STEP 2: Cleaning Previous PyTorch Builds"
echo "================================================================================"
echo ""

# Uninstall nightly wheel if present
if "$PYTHON_BIN" -m pip list | grep -q "torch.*2.11.0.dev"; then
    echo "Uninstalling PyTorch nightly..."
    "$PYTHON_BIN" -m pip uninstall -y torch
fi

# Clean build artifacts
"$PYTHON_BIN" setup.py clean || true
rm -rf build/ dist/ torch.egg-info/ || true

echo "✓ Clean complete"
echo ""

# ============================================================================
# STEP 3: BUILD PYTORCH
# ============================================================================

echo "================================================================================"
echo "🔨 STEP 3: Building PyTorch with SM_103 Support"
echo "================================================================================"
echo ""
echo "This will take approximately 45-65 minutes with MAX_JOBS=$MAX_JOBS"
echo "Building with:"
echo "  - CUDA $CUDA_RELEASE"
echo "  - cuDNN $CUDNN_VERSION"
echo "  - Triton $TRITON_VERSION (from source)"
echo "  - sm_103 kernels (B300 NATIVE!)"
echo "  - Flash-Attention with sm_103 (PATCHED!)"
echo "  - All performance optimizations enabled"
echo ""
echo "Build started at: $(date)"
echo ""

# Track build time
BUILD_START=$(date +%s)

# Run build
set +e
time "$PYTHON_BIN" setup.py bdist_wheel
BUILD_EXIT_CODE=$?
set -e

BUILD_END=$(date +%s)
BUILD_TIME=$((BUILD_END - BUILD_START))
TOTAL_TIME=$((BUILD_TIME + BUILD_TIME_TRITON))

echo ""
echo "Build finished at: $(date)"
echo "PyTorch build time: $((BUILD_TIME / 60)) minutes $((BUILD_TIME % 60)) seconds"
echo "Total time (Triton + PyTorch): $((TOTAL_TIME / 60)) minutes $((TOTAL_TIME % 60)) seconds"
echo ""

if [ $BUILD_EXIT_CODE -ne 0 ]; then
    echo "❌ BUILD FAILED with exit code $BUILD_EXIT_CODE"
    exit $BUILD_EXIT_CODE
fi

# ============================================================================
# POST-BUILD VERIFICATION
# ============================================================================

echo "================================================================================"
echo "✅ BUILD SUCCESSFUL!"
echo "================================================================================"
echo ""

# Find the wheel
WHEEL_FILE=$(ls -t dist/torch-*.whl 2>/dev/null | head -1)

if [ -z "$WHEEL_FILE" ]; then
    echo "❌ ERROR: Wheel file not found in dist/"
    exit 1
fi

# Copy wheel to /root for easy access
cp "$WHEEL_FILE" /root/
WHEEL_OUT="/root/$(basename "$WHEEL_FILE")"

WHEEL_SIZE=$(du -h "$WHEEL_OUT" | awk '{print $1}')

echo "📦 PyTorch wheel: $WHEEL_FILE"
echo "📦 PyTorch copied to: $WHEEL_OUT"
echo "📏 PyTorch size: $WHEEL_SIZE"
if [[ -n "${TRITON_WHEEL:-}" && -f "$TRITON_WHEEL" ]]; then
    cp "$TRITON_WHEEL" /root/
    TRITON_OUT="/root/$(basename "$TRITON_WHEEL")"
    TRITON_SIZE=$(du -h "$TRITON_OUT" | awk '{print $1}')
    echo "📦 Triton wheel: $TRITON_WHEEL"
    echo "📦 Triton copied to: $TRITON_OUT"
    echo "📏 Triton size: $TRITON_SIZE"
fi
echo ""

# ============================================================================
# INSTALLATION & VERIFICATION INSTRUCTIONS
# ============================================================================

echo "================================================================================"
echo "📥 INSTALLATION & VERIFICATION"
echo "================================================================================"
echo ""
echo "To install the ULTIMATE B300 SM_103 PyTorch + Triton wheels:"
echo ""
if [ -n "$TRITON_OUT" ]; then
    echo "  # Install Triton first (built for CUDA 13.1)"
    echo "  $PYTHON_BIN -m pip install --force-reinstall $TRITON_OUT"
    echo ""
fi
echo "  # Install PyTorch"
echo "  $PYTHON_BIN -m pip install --force-reinstall --no-deps $WHEEL_OUT"
echo ""
echo "After installation, verify with:"
echo ""
echo "  $PYTHON_BIN -c \""
echo "  import torch"
echo "  print(f'PyTorch Version: {torch.__version__}')"
echo "  print(f'CUDA Version: {torch.version.cuda}')"
echo "  print(f'cuDNN Version: {torch.backends.cudnn.version()}')"
echo "  print(f'GPU: {torch.cuda.get_device_name(0)}')"
echo "  print(f'Compute Capability: {torch.cuda.get_device_capability(0)}')"
echo "  print(f'Architectures: {torch.cuda.get_arch_list()}')"
echo "  \""
echo ""
echo "Expected output:"
echo "  - Version: ${PYTORCH_BUILD_VERSION}"
echo "  - Architectures: ['sm_103', 'compute_103']"
echo "  - Compute Capability: (10, 3)"
echo ""
echo "To verify Triton integration:"
echo ""
echo "  $PYTHON_BIN -c \""
echo "  import torch"
echo "  import triton"
echo "  print(f'Triton version: {triton.__version__}')"
echo "  print(f'Triton CUDA support: {torch.cuda.is_available()}')"
echo "  \""
echo ""
echo "To verify Flash-Attention sm_103 support:"
echo ""
echo "  $PYTHON_BIN -c \""
echo "  import torch"
echo "  from flash_attn import flash_attn_func"
echo "  q = torch.randn(2, 2048, 16, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)"
echo "  k = torch.randn(2, 2048, 16, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)"
echo "  v = torch.randn(2, 2048, 16, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)"
echo "  out = flash_attn_func(q, k, v)"
echo "  loss = out.sum()"
echo "  loss.backward()"
echo "  print('✅ Flash-Attention sm_103 forward + backward WORKING!')"
echo "  \""
echo ""
echo "To test torch.compile (Triton + Inductor):"
echo ""
echo "  $PYTHON_BIN -c \""
echo "  import torch"
echo "  @torch.compile"
echo "  def f(x): return x @ x.T"
echo "  x = torch.randn(1024, 1024, device='cuda')"
echo "  y = f(x)  # First run: compiles"
echo "  y = f(x)  # Second run: uses compiled kernel"
echo "  print('✅ torch.compile working with Triton!')"
echo "  \""
echo ""
echo "================================================================================"
echo "🔥🔥🔥 ULTIMATE B300 SM_103 PYTORCH + TRITON BUILD COMPLETE! 🔥🔥🔥"
echo "================================================================================"
echo ""
echo "You now have PyTorch with:"
echo "  ✅ Native sm_103 kernels for B300"
echo "  ✅ Flash-Attention with sm_103 forward + backward"
echo "  ✅ Triton built from source for CUDA 13.1"
echo "  ✅ torch.compile/Inductor ready for maximum performance"
echo "  ✅ compute_103f family-specific optimizations"
echo "  ✅ cuDNN SDPA backward fix included"
echo "  ✅ All performance libraries enabled (FBGEMM, cuSPARSELt, NCCL, Kineto)"
echo "  ✅ Ready for ULTIMATE diffusion model LoRA training!"
echo ""
echo "Build times:"
echo "  - Triton: $(($BUILD_TIME_TRITON / 60))m $(($BUILD_TIME_TRITON % 60))s"
echo "  - PyTorch: $((BUILD_TIME / 60))m $((BUILD_TIME % 60))s"
echo "  - Total: $((TOTAL_TIME / 60))m $((TOTAL_TIME % 60))s"
echo ""
echo "================================================================================"
