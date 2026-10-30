# NCCL Send/Receive Performance Test

This repository builds only `sendrecv_perf`. Each rank sends to the next rank
and receives from the previous rank in a ring. The shared timing, validation,
profiling, and JSON output code is retained.

## Build

Requirements: Linux, Bash, a CUDA toolkit with `nvcc`, a compatible C++ compiler,
and NCCL headers and libraries. MPI is optional for multi-process/multi-node runs.
Make and CMake are not used.

```sh
CUDA_HOME=/usr/local/cuda NCCL_HOME=/path/to/nccl ./build.sh
```

The executable is `build/sendrecv_perf`. Each invocation recompiles all required
sources. If NCCL is installed in the compiler's default search paths, omit
`NCCL_HOME`. It should otherwise contain `include/nccl.h` and `lib` or `lib64`.

Environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `CUDA_HOME` | `/usr/local/cuda` | CUDA installation |
| `NVCC` | `$CUDA_HOME/bin/nvcc` | CUDA compiler executable |
| `CXX` | `g++` | Host C++ compiler executable |
| `CUDA_LIB` | `$CUDA_HOME/lib64` | CUDA library directory |
| `NCCL_HOME` | unset | NCCL installation |
| `BUILDDIR` | repository `build/` | Output directory; relative paths use the current directory |
| `CUDA_ARCHS` | selected by CUDA version | Space-separated GPU architecture numbers, e.g. `80 90` |
| `MPI` | `0` | Set to `1` to enable MPI |
| `MPI_HOME` | unset | MPI installation with `include` and `lib` or `lib64`; required for `MPI=1` |

To build for a specific GPU architecture:

```sh
CUDA_ARCHS=90 NCCL_HOME=/path/to/nccl ./build.sh
```

The script also emits PTX for the last architecture in the list.

## Run

Ensure CUDA and NCCL shared libraries are available to the runtime loader. For a
custom installation, for example:

```sh
export LD_LIBRARY_PATH=/path/to/nccl/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
./build/sendrecv_perf -b 8 -e 128M -f 2 -g 2
./build/sendrecv_perf --help
```

Use `lib64` instead of `lib` if that is where NCCL is installed.
The example runs on two local GPUs.

For MPI (one GPU per process):

```sh
MPI=1 MPI_HOME=/path/to/mpi NCCL_HOME=/path/to/nccl ./build.sh
/path/to/mpi/bin/mpirun -np 2 ./build/sendrecv_perf -b 8 -e 128M -f 2 -g 1
```

For multiple nodes, deploy the executable and libraries to each node and configure
your MPI launcher and GPU visibility for the cluster. MPI libraries must also be
available to the runtime loader.

See [performance metrics](doc/PERFORMANCE.md) for bandwidth definitions.
`tools/analyze_perf_json.py` can analyze per-iteration JSON output generated with
`-I 1 -J results.json`.
