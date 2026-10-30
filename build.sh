#!/usr/bin/env bash
# Build the sample on Linux without Make or CMake.
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
usage() {
  echo 'Usage: [CUDA_HOME=...] [NCCL_HOME=...] MPI_HOME=... ./build.sh'
}
if [[ $# -gt 0 ]]; then
  for target in "$@"; do
    case "$target" in
      -h|--help) usage; exit 0 ;;
      *) echo "Unknown build target: $target" >&2; usage >&2; exit 1 ;;
    esac
  done
fi
[[ $(uname -s) == Linux ]] || { echo 'build.sh requires Linux.' >&2; exit 1; }
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
NVCC=${NVCC:-$CUDA_HOME/bin/nvcc}
CXX=${CXX:-g++}
BUILDDIR=${BUILDDIR:-$root/build}
MPI=${MPI:-0}
[[ $MPI == 1 ]] || { echo 'Building sample requires MPI=1.' >&2; exit 1; }
for compiler in "$NVCC" "$CXX"; do
  command -v "$compiler" >/dev/null || { echo "Compiler not found: $compiler" >&2; exit 1; }
done
version=$("$NVCC" --version)
[[ $version =~ release[[:space:]]+([0-9]+)\.([0-9]+) ]] || {
  echo 'Cannot determine CUDA version.' >&2; exit 1;
}
major=${BASH_REMATCH[1]}
minor=${BASH_REMATCH[2]}
standard=c++14
if (( major >= 13 )); then
  standard=c++17
  architectures=(75 80 90 100 120)
elif (( major == 12 && minor >= 8 )); then
  architectures=(60 61 70 80 90 100 120)
elif (( major >= 12 )); then
  architectures=(60 61 70 80 90)
elif (( major >= 11 )); then
  architectures=(60 61 70 80)
else
  architectures=(35 50 60 61 70)
fi
# Space-separated architecture numbers; emit PTX for the final entry as well.
if [[ -n ${CUDA_ARCHS:-} ]]; then
  read -r -a architectures <<< "$CUDA_ARCHS"
fi
gencode=()
for arch in "${architectures[@]}"; do
  [[ $arch =~ ^[0-9]+[af]?$ ]] || { echo "Invalid CUDA architecture: $arch" >&2; exit 1; }
  gencode+=("-gencode=arch=compute_$arch,code=sm_$arch")
done
gencode+=("-gencode=arch=compute_$arch,code=compute_$arch")
flags=("-std=$standard" -O3 -g -DNCCL_OS_LINUX -DNCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE=1
       "-I$root/src" "-I$root/os" "-I$root/verifiable" "-I$BUILDDIR/obj/include")
libs=("-L${CUDA_LIB:-$CUDA_HOME/lib64}" -lcudart -lnccl -lrt -ldl -lpthread)
if [[ -n ${NCCL_HOME:-} ]]; then
  flags+=("-I$NCCL_HOME/include")
  libs=("-L$NCCL_HOME/lib" "-L$NCCL_HOME/lib64" "${libs[@]}")
fi
if [[ $MPI == 1 ]]; then
  : "${MPI_HOME:?Set MPI_HOME to your MPI installation for MPI=1}"
  flags+=(-DMPI_SUPPORT "-I$MPI_HOME/include")
  libs+=("-L$MPI_HOME/lib" "-L$MPI_HOME/lib64" -lmpi)
fi
printf 'Compiling src/sample.cu\n'
"$NVCC" -ccbin "$CXX" "${flags[@]}" "${gencode[@]}" -Xcompiler -pthread \
  "$root/src/sample.cu" "${libs[@]}" -o "$BUILDDIR/sample"
printf 'Built %s/sample\n' "$BUILDDIR"
