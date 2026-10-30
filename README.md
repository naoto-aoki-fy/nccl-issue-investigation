# NCCL Send/Receive Performance Test

This repository builds a standalone MPI sample from `src/sample.cu`. Each rank
sends to the next rank and receives from the previous rank in a ring using NCCL
and one GPU per process. The sample reports CPU and/or CUDA event timing and
bandwidth, then validates the received data. Progress and results are written to
standard error; JSON output is not implemented.

## Build

Requirements: Linux, Bash, a CUDA toolkit with `nvcc`, a compatible C++ compiler,
and NCCL and MPI headers and libraries. MPI must support
`MPI_Comm_split_type` and `MPI_COMM_TYPE_SHARED`.
Make and CMake are not used.

Run the following from the repository root:

```sh
mkdir -p build
CUDA_HOME=/usr/local/cuda NCCL_HOME=/path/to/nccl MPI=1 MPI_HOME=/path/to/mpi ./build.sh
```

The script builds only `sample` in `BUILDDIR` (by default, `build/` in the
repository). It does not create the output directory, so create it before
building, including when overriding `BUILDDIR`. Do not pass `sample` as a
command-line target. `MPI=1` and `MPI_HOME` are required; leaving `MPI` unset or
setting it to `0` causes an error. Use `./build.sh --help` for usage.

If NCCL is installed in the
compiler's default search paths, omit `NCCL_HOME`. It should otherwise contain
`include/nccl.h` and `lib` or `lib64`.

Environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `CUDA_HOME` | `/usr/local/cuda` | CUDA installation |
| `NVCC` | `$CUDA_HOME/bin/nvcc` | CUDA compiler executable |
| `CXX` | `g++` | Host C++ compiler executable |
| `CUDA_LIB` | `$CUDA_HOME/lib64` | CUDA library directory |
| `NCCL_HOME` | unset | NCCL installation |
| `BUILDDIR` | repository `build/` | Existing output directory; relative paths use the current directory |
| `CUDA_ARCHS` | selected by CUDA version | Space-separated GPU architecture numbers, e.g. `80 90` |
| `MPI` | `0` | Must be set to `1` |
| `MPI_HOME` | unset | MPI installation with `include` and `lib` or `lib64`; required for `MPI=1` |

To build for a specific GPU architecture:

```sh
mkdir -p build
CUDA_ARCHS=90 MPI=1 MPI_HOME=/path/to/mpi NCCL_HOME=/path/to/nccl ./build.sh
```

The script also emits PTX for the last architecture in the list.

## Run

Ensure CUDA, NCCL, and MPI shared libraries are available to the runtime loader. For a
custom installation, for example:

```sh
export LD_LIBRARY_PATH=/path/to/mpi/lib:/path/to/nccl/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
/path/to/mpi/bin/mpirun -np 2 ./build/sample -s 4M -w 1 -n 20
/path/to/mpi/bin/mpirun -np 1 ./build/sample -h
```

Use `lib64` instead of `lib` where appropriate for the MPI and NCCL installations.
The measurement example uses two MPI ranks and, when run on one node, two GPUs.
Help uses `-h` and exits before checking the GPU or rank count requirements.

Measurement requires at least two MPI ranks in total. Each process selects CUDA
device `localRank`, where the local rank is determined by an MPI shared-memory
communicator. Each process must see at least as many CUDA devices as there are
local ranks. On each node, expose the same ordered GPU list to all local ranks;
masking each process to a single GPU does not work with multiple local ranks.

For multiple nodes, deploy the executable and libraries to each node and configure
your MPI launcher and GPU visibility for the cluster.

## Buffer and iteration options

| Option | Default | Purpose |
| --- | --- | --- |
| `-s SIZE` | `4M` | Bytes sent and received per rank per measured iteration. |
| `-n N` | `20` | Measured iterations; must be positive. |
| `-w N` | `1` | Warmup iterations; `0` disables warmup transfers. |
| `-W SIZE` | same as `-s` | Bytes sent and received per warmup iteration. |
| `-A TYPE` | `nccl` | Buffer allocator: `nccl` or `cuda`. |
| `-h` | — | Show help and exit. |

Sizes are positive byte counts divisible by 4. Optional `K`, `M`, and `G`
suffixes (also lowercase) use powers of 1024. Transfers use `ncclFloat`.
Each rank allocates separate GPU send and receive buffers, each sized to the
larger of `-s` and `-W`, even when `-w 0` is used.

The default NCCL buffer allocator requires NCCL 2.19+ headers and a compatible
library. Use `-A cuda` to allocate buffers with CUDA instead.

## Timing and results

The sample defaults to 20 consecutive measured iterations (`-n`), with one
send/receive group per iteration and a single completion wait after the loop.
It measures CPU wall-clock time from the first submission through GPU completion,
averages the per-iteration times across ranks, and reports one-direction TX
bandwidth in decimal GB/s. By default, it also reports each rank's CPU time and
TX/TX+RX bandwidth. Warmup, MPI barriers, logging, host copies, and data
validation are outside the measured region. `-w` controls only warmup iterations;
`-W` selects a separate warmup size and otherwise defaults to `-s`.

For a per-iteration time `t` in seconds, TX bandwidth is `size / t / 1e9` GB/s;
the local TX+RX figure is twice that value. Rank-average TX bandwidth uses the
average time across ranks, rather than averaging their bandwidths.

After measurement, each rank checks that every received float equals the previous
rank's number. A successful run reports `Global validation result: OK` and exits
with status 0; a validation failure exits with status 1.

## Communicator, synchronization, and reporting options

The following sample options allow one change at a time to be compared. The first
value shown is the default; `default` in a communicator field leaves that field
at the NCCL library default. Explicit settings require the listed NCCL headers.
Environment variables may override communicator settings.

| Option | Values | Purpose |
| --- | --- | --- |
| `--comm-init` | `config`, `rank` | Initialize with `ncclCommInitRankConfig` or `ncclCommInitRank` (library defaults). Config requires NCCL 2.14+; older headers use rank initialization. |
| `--max-p2p-peers` | `2`, `default`, or `1..nranks` | Limit concurrent P2P peers (NCCL 2.30+). |
| `--nvlink-centric-sched` | `1`, `default`, `0` | Set NVLink-centric scheduling (NCCL 2.28+). |
| `--graph-usage-mode` | `0`, `default`, `1`, `2` | Set communicator graph usage mode (NCCL 2.29+). |
| `--stream` | `nonblocking`, `blocking` | Set CUDA stream flags; `blocking` creates a regular stream. |
| `--wait` | `poll`, `stream`, `event` | Wait after measurement via stream query and yield, `cudaStreamSynchronize`, or `cudaEventSynchronize`. Event wait records a stop event even in CPU timing mode. |
| `--blocking-sync` | `0`, `1` | When set to 1, use `cudaDeviceScheduleBlockingSync` with `--wait stream` or create the stop event with `cudaEventBlockingSync` for `--wait event`, allowing the CPU thread to sleep during synchronization. Event timing remains enabled. |
| `--poll-yield` | `1`, `0` | Enable or disable `std::this_thread::yield()` in polling waits, including warmup. Async error checks remain enabled. Has no effect on stream/event waits. |
| `--warmup-wait` | `poll`, `stream` | Select the completion wait after warmup. |
| `--pre-log` | `before`, `after` | Place the measurement-start log before or after the measurement barrier. The log is outside the local CPU timing interval. |
| `--timing` | `cpu`, `event`, `both` | Report CPU wall-clock time, CUDA event time, or both. Event time spans all measured iterations and is divided by `-n`. |
| `--rank-report` | `both`, `average`, `per-rank` | Print rank-average timing, per-rank timing and TX/TX+RX bandwidth, or both. |

`--comm-init rank` uses NCCL's defaults and cannot be combined with explicit
communicator field options. On older NCCL headers, unsupported fields use the
library default; explicitly setting an unsupported field to a non-default value
is an error. When CUDA event timing is enabled, recording events also adds work
to the CPU-timed region. A stop event used for `--wait event` similarly adds
instrumentation even with `--timing cpu`.

For example, compare the baseline with individual changes (use the same MPI
launcher, placement, and environment for every run):

```sh
mpirun -np 2 ./build/sample -s 4M -w 20 -n 1
mpirun -np 2 ./build/sample -s 4M -w 20 -n 1 --wait stream
mpirun -np 2 ./build/sample -s 4M -w 20 -n 1 --wait event --blocking-sync 0
mpirun -np 2 ./build/sample -s 4M -w 20 -n 1 --wait event --blocking-sync 1
mpirun -np 2 ./build/sample -s 4M -w 20 -n 1 --wait poll --poll-yield 0
mpirun -np 2 ./build/sample -s 4M -w 20 -n 1 --pre-log after
mpirun -np 2 ./build/sample -s 4M -w 20 -n 1 --max-p2p-peers default
mpirun -np 2 ./build/sample -s 4M -w 20 -n 1 --timing both
```

Use the same rank count, host placement, rank ordering, CPU binding, visible GPUs,
NCCL/CUDA libraries, and environment variables when comparing runs. The examples
above assume `mpirun` is on `PATH` and use one measured iteration (`-n 1`) to
compare individual settings; omit that override for the default 20 iterations.
