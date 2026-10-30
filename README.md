# NCCL Send/Receive Performance Test

This repository builds `sendrecv_perf` and, with MPI enabled, `sample` from
`src/sample.cu`. Each rank sends to the next rank
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
| `MPI` | `0` | Set to `1` to enable MPI and also build `sample` |
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

The same build also produces `build/sample`, a standalone send/receive sample
that requires at least two MPI ranks and one GPU per local rank:

```sh
/path/to/mpi/bin/mpirun -np 2 ./build/sample -s 4M -w 1 -n 20
```

The sample defaults to NCCL buffer allocation (NCCL 2.19+); use `-A cuda` to
allocate buffers with CUDA instead. With `MPI=0`, the sample is skipped.

The sample defaults to 20 consecutive measured iterations (`-n`), with one
send/receive group per iteration and a single completion wait after the loop.
It measures CPU wall-clock time from the first submission through GPU completion,
averages the per-iteration times across ranks, and reports one-direction TX
bandwidth in decimal GB/s. Warmup, MPI barriers, logging, host copies, and data
validation are outside the measured region. `-w` controls only warmup iterations;
`-W` selects a separate warmup size and otherwise defaults to `-s`.

To compare both programs at 4 MiB with 20 warmup and 20 measured iterations:

```sh
/path/to/mpi/bin/mpirun -np 2 ./build/sample -s 4M -w 20 -n 20 -A nccl
/path/to/mpi/bin/mpirun -np 2 ./build/sendrecv_perf \
  -b 4M -e 4M -g 1 -d float -w 20 -n 20 -m 1 -z 0 -G 0 -C 0 -R 0 -a 1
```

Use the same rank count, host placement, rank ordering, CPU binding, visible GPUs,
NCCL/CUDA libraries, and environment variables for both commands. The sample uses
the same nonblocking stream and normal non-graph communicator settings as
`sendrecv_perf`, including `maxP2pPeers = 2` with NCCL 2.30+ (also relevant to
multi-node InfiniBand communication). Compare its `Rank-average CPU time` and
`TX bandwidth` with the benchmark's `time` and `algbw`/`busbw` columns.

For multiple nodes, deploy the executable and libraries to each node and configure
your MPI launcher and GPU visibility for the cluster. MPI libraries must also be
available to the runtime loader.

See [performance metrics](doc/PERFORMANCE.md) for bandwidth definitions.
`tools/analyze_perf_json.py` can analyze per-iteration JSON output generated with
`-I 1 -J results.json`.
