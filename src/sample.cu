#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>

#include <unistd.h>       // getopt
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>
#include <string>
#include <thread>
#include <vector>

static int g_rank = -1;
static int g_gpu = -1;
static std::chrono::steady_clock::time_point g_start;

// CPU-side progress log; no logging is performed during the benchmark.
static void log_message(const char* fmt, ...) {
    double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - g_start).count();

    fprintf(stderr, "[%8.3f s][rank %d][gpu %d] ",
            elapsed, g_rank, g_gpu);

    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);

    fputc('\n', stderr);
    fflush(stderr);
}

#define LOG(...) log_message(__VA_ARGS__)

#define MPICHECK(cmd) do {                                           \
    int e = (cmd);                                                   \
    if (e != MPI_SUCCESS) {                                          \
        LOG("ERROR: MPI at %s:%d code=%d", __FILE__, __LINE__, e);  \
        MPI_Abort(MPI_COMM_WORLD, 1);                                \
    }                                                                \
} while (0)

#define CUDACHECK(cmd) do {                                          \
    cudaError_t e = (cmd);                                           \
    if (e != cudaSuccess) {                                          \
        LOG("ERROR: CUDA %s at %s:%d",                               \
            cudaGetErrorString(e), __FILE__, __LINE__);              \
        MPI_Abort(MPI_COMM_WORLD, 1);                                \
    }                                                                \
} while (0)

#define NCCLCHECK(cmd) do {                                          \
    ncclResult_t e = (cmd);                                          \
    if (e != ncclSuccess) {                                          \
        LOG("ERROR: NCCL %s at %s:%d",                               \
            ncclGetErrorString(e), __FILE__, __LINE__);              \
        MPI_Abort(MPI_COMM_WORLD, 1);                                \
    }                                                                \
} while (0)

// SIZE is bytes, optionally followed by K/M/G (powers of 1024).
// ncclFloat requires a multiple of sizeof(float) bytes.
static bool parse_buffer_size(const char* text, size_t* bytes) {
    if (text == nullptr || text[0] < '0' || text[0] > '9') {
        return false;
    }

    errno = 0;
    char* end = nullptr;
    unsigned long long n = std::strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || n == 0) {
        return false;
    }

    unsigned long long factor = 1;
    if (*end != '\0') {
        if (end[1] != '\0') return false;
        switch (*end) {
        case 'K': case 'k': factor = 1024ULL; break;
        case 'M': case 'm': factor = 1024ULL * 1024; break;
        case 'G': case 'g': factor = 1024ULL * 1024 * 1024; break;
        default: return false;
        }
    }

    if (n > ULLONG_MAX / factor) return false;
    unsigned long long total = n * factor;
    if (total > std::numeric_limits<size_t>::max()) return false;
    if (total % sizeof(float) != 0) return false;

    *bytes = static_cast<size_t>(total);
    return true;
}

// Iteration counts are nonnegative; measured iterations must also be nonzero.
static bool parse_iteration_count(const char* text, int* count) {
    if (!text || !*text || *text < '0' || *text > '9') return false;
    errno = 0;
    char* end = nullptr;
    long n = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || n < 0 || n > INT_MAX)
        return false;
    *count = static_cast<int>(n);
    return true;
}

// Match sendrecv_perf's completion polling, including async error checks and
// yielding the CPU so NCCL proxy threads can make progress.
static void wait_for_transfer(cudaStream_t stream, ncclComm_t comm) {
    for (;;) {
        cudaError_t status = cudaStreamQuery(stream);
        if (status == cudaSuccess) return;
        if (status != cudaErrorNotReady) CUDACHECK(status);
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 4, 0)
        ncclResult_t async_error;
        NCCLCHECK(ncclCommGetAsyncError(comm, &async_error));
        NCCLCHECK(async_error);
#endif
        std::this_thread::yield();
    }
}

static void usage(const char* program) {
    fprintf(stderr,
        "Usage: %s [-s SIZE] [-n ITERS] [-w WARMUP] [-W SIZE] [-A nccl|cuda] [-h]\n"
        "  -s SIZE  Size of EACH GPU send/recv buffer (default: 4M)\n"
        "           No suffix: bytes; K/M/G: powers of 1024\n"
        "           Examples: -s 4096, -s 64K, -s 4M, -s 1G\n"
        "           Must be positive and divisible by sizeof(float)=4\n"
        "  -n N     Measured iterations (default: 20; must be positive)\n"
        "  -w N     Untimed warmup iterations (default: 1; 0 disables)\n"
        "  -W SIZE  Size of EACH warmup send/recv (default: same as -s)\n"
        "  -A TYPE  GPU buffer allocator: nccl (default) or cuda\n"
        "  -h       Show help\n",
        program);
}

int main(int argc, char** argv) {
    fprintf(stderr, "[startup] MPI_Init begin\n");
    fflush(stderr);

    int init_err = MPI_Init(&argc, &argv);
    if (init_err != MPI_SUCCESS) {
        fprintf(stderr, "ERROR: MPI_Init failed (code=%d)\n", init_err);
        return 1;
    }
    g_start = std::chrono::steady_clock::now();

    int rank = -1, nranks = 0;
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nranks));
    g_rank = rank;
    LOG("MPI initialized: nranks=%d", nranks);

    // Every rank parses arguments and exits collectively on a CLI error or -h.
    size_t bytes = 4ULL * 1024 * 1024;
    int iters = 20;
    int warmup_iters = 1;
    size_t warmup_bytes = 0;
    bool warmup_size_specified = false;
    bool show_help = false;
    bool use_nccl_alloc = true;
    std::string parse_error;

    opterr = 0;
    int opt;
    while ((opt = getopt(argc, argv, "s:n:w:W:A:h")) != -1) {
        switch (opt) {
        case 's':
            if (!parse_buffer_size(optarg, &bytes)) {
                parse_error = "invalid -s SIZE (positive, multiple of 4, optional K/M/G)";
            }
            break;
        case 'n':
            if (!parse_iteration_count(optarg, &iters) || iters == 0) {
                parse_error = "invalid -n N (positive integer)";
            }
            break;
        case 'w':
            if (!parse_iteration_count(optarg, &warmup_iters)) {
                parse_error = "invalid -w N (nonnegative integer)";
            }
            break;
        case 'W':
            if (!parse_buffer_size(optarg, &warmup_bytes)) {
                parse_error = "invalid -W SIZE (positive, multiple of 4, optional K/M/G)";
            } else {
                warmup_size_specified = true;
            }
            break;
        case 'A':
            if (std::string(optarg) == "nccl") {
                use_nccl_alloc = true;
            } else if (std::string(optarg) == "cuda") {
                use_nccl_alloc = false;
            } else {
                parse_error = "invalid -A TYPE (nccl or cuda)";
            }
            break;
        case 'h':
            show_help = true;
            break;
        default:
            parse_error = "unknown option or missing option argument";
            break;
        }
    }
    if (optind < argc) parse_error = "unexpected positional arguments";

    int local_error = parse_error.empty() ? 0 : 1;
    int any_error = 0;
    MPICHECK(MPI_Allreduce(&local_error, &any_error, 1, MPI_INT,
                           MPI_MAX, MPI_COMM_WORLD));
    if (any_error) {
        if (local_error) LOG("ERROR: %s", parse_error.c_str());
        if (rank == 0) usage(argv[0]);
        MPICHECK(MPI_Finalize());
        return 2;
    }

    int local_help = show_help ? 1 : 0;
    int any_help = 0;
    MPICHECK(MPI_Allreduce(&local_help, &any_help, 1, MPI_INT,
                           MPI_MAX, MPI_COMM_WORLD));
    if (any_help) {
        if (rank == 0) usage(argv[0]);
        MPICHECK(MPI_Finalize());
        return 0;
    }

#if NCCL_VERSION_CODE < NCCL_VERSION(2, 19, 0)
    if (use_nccl_alloc) {
        if (rank == 0) LOG("ERROR: -A nccl requires NCCL 2.19+ headers");
        MPICHECK(MPI_Finalize());
        return 2;
    }
#endif

    if (nranks < 2) {
        LOG("ERROR: requires at least 2 ranks");
        MPICHECK(MPI_Finalize());
        return 1;
    }

    if (!warmup_size_specified) warmup_bytes = bytes;
    const size_t count = bytes / sizeof(float);
    const size_t warmup_count = warmup_bytes / sizeof(float);
    const size_t allocation_bytes = (bytes > warmup_bytes) ? bytes : warmup_bytes;
    const size_t allocation_count = allocation_bytes / sizeof(float);
    constexpr double MiB = 1024.0 * 1024.0;
    LOG("Configuration: buffer=%zu bytes (%.2f MiB), elements=%zu float",
        bytes, static_cast<double>(bytes) / MiB, count);
    LOG("Measurement configuration: %d consecutive iterations, CPU wall-clock timing", iters);
    LOG("Warmup configuration: %d iterations, %zu bytes (%.2f MiB), elements=%zu float",
        warmup_iters, warmup_bytes, static_cast<double>(warmup_bytes) / MiB, warmup_count);

    LOG("Creating local MPI communicator");
    MPI_Comm localComm;
    MPICHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
                                 rank, MPI_INFO_NULL, &localComm));
    int localRank = -1, localSize = 0;
    MPICHECK(MPI_Comm_rank(localComm, &localRank));
    MPICHECK(MPI_Comm_size(localComm, &localSize));
    LOG("Local communicator ready: localRank=%d localSize=%d",
        localRank, localSize);

    LOG("Querying CUDA devices");
    int deviceCount = 0;
    CUDACHECK(cudaGetDeviceCount(&deviceCount));
    if (localSize > deviceCount) {
        LOG("ERROR: localSize=%d > deviceCount=%d", localSize, deviceCount);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDACHECK(cudaSetDevice(localRank));
    g_gpu = localRank;

    cudaDeviceProp prop;
    CUDACHECK(cudaGetDeviceProperties(&prop, localRank));
    LOG("GPU selected: %s", prop.name);

    ncclUniqueId id;
    if (rank == 0) {
        LOG("Generating NCCL unique ID");
        NCCLCHECK(ncclGetUniqueId(&id));
    }
    LOG("Broadcasting NCCL unique ID");
    MPICHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

    LOG("Initializing NCCL communicator");
    ncclComm_t comm;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 14, 0)
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    // Keep the normal, non-graph sendrecv_perf communicator settings.
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
    config.nvlinkCentricSched = 1;
#endif
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 0)
    config.graphUsageMode = 0;
#endif
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 0)
    config.maxP2pPeers = 2;
#endif
    NCCLCHECK(ncclCommInitRankConfig(&comm, nranks, id, rank, &config));
#else
    NCCLCHECK(ncclCommInitRank(&comm, nranks, id, rank));
#endif
    LOG("NCCL communicator initialized");

    LOG("Creating nonblocking CUDA stream");
    cudaStream_t stream;
    CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    float *d_send = nullptr, *d_recv = nullptr;
    // One pair of buffers is reused; allocate enough for either transfer size.
    LOG("Allocating GPU buffers: %.2f MiB each (total %.2f MiB)",
        static_cast<double>(allocation_bytes) / MiB,
        2.0 * static_cast<double>(allocation_bytes) / MiB);
    LOG("GPU buffer allocator: %s", use_nccl_alloc ? "ncclMemAlloc" : "cudaMalloc");
    if (use_nccl_alloc) {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 19, 0)
        NCCLCHECK(ncclMemAlloc(reinterpret_cast<void**>(&d_send), allocation_bytes));
        NCCLCHECK(ncclMemAlloc(reinterpret_cast<void**>(&d_recv), allocation_bytes));
#endif
    } else {
        CUDACHECK(cudaMalloc(&d_send, allocation_bytes));
        CUDACHECK(cudaMalloc(&d_recv, allocation_bytes));
    }
    LOG("GPU buffers allocated");

    LOG("Initializing host buffers and send data");
    std::vector<float> h_send, h_recv;
    try {
        h_send.assign(allocation_count, float(rank));
        h_recv.assign(count, -1.0f);
    } catch (const std::bad_alloc&) {
        LOG("ERROR: host buffer allocation failed");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDACHECK(cudaMemcpy(d_send, h_send.data(), allocation_bytes,
                         cudaMemcpyHostToDevice));
    LOG("Send buffer initialized");

    int sendPeer = (rank + 1) % nranks;
    int recvPeer = (rank - 1 + nranks) % nranks;
    LOG("Communication plan: sendPeer=%d recvPeer=%d", sendPeer, recvPeer);

    LOG("Waiting at MPI barrier");
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    LOG("MPI barrier completed");

    // Run the same communication as the timed path, outside the timed region.
    // This allows NCCL to establish P2P connections and initialize kernels.
    LOG("Warming up NCCL send/recv: %d iterations, %.2f MiB each direction",
        warmup_iters, static_cast<double>(warmup_bytes) / MiB);
    for (int i = 0; i < warmup_iters; ++i) {
        NCCLCHECK(ncclGroupStart());
        NCCLCHECK(ncclSend(d_send, warmup_count, ncclFloat, sendPeer, comm, stream));
        NCCLCHECK(ncclRecv(d_recv, warmup_count, ncclFloat, recvPeer, comm, stream));
        NCCLCHECK(ncclGroupEnd());
    }
    // Ensure ALL warmup GPU work has finished before starting timing.
    wait_for_transfer(stream, comm);
    LOG("Starting %d consecutive NCCL transfers: %.2f MiB TX + %.2f MiB RX each",
        iters,
        static_cast<double>(bytes) / MiB,
        static_cast<double>(bytes) / MiB);

    // Log before this barrier to avoid rank skew from progress output.
    // Match sendrecv_perf -m 1 -z 0 -G 0: one group per iteration,
    // no intermediate synchronization, and one completion wait at the end.
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto measurement_start = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        NCCLCHECK(ncclGroupStart());
        NCCLCHECK(ncclSend(d_send, count, ncclFloat, sendPeer, comm, stream));
        NCCLCHECK(ncclRecv(d_recv, count, ncclFloat, recvPeer, comm, stream));
        NCCLCHECK(ncclGroupEnd());
    }
    wait_for_transfer(stream, comm);
    double local_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - measurement_start).count() / iters;

    // Match sendrecv_perf -a 1: average the per-iteration times across ranks,
    // then calculate bandwidth using one sent message (not TX+RX).
    double sum_seconds = 0.0;
    MPICHECK(MPI_Allreduce(&local_seconds, &sum_seconds, 1, MPI_DOUBLE,
                           MPI_SUM, MPI_COMM_WORLD));
    double average_seconds = sum_seconds / nranks;
    LOG("Local CPU time: %.3f us/iteration (%d iterations)",
        local_seconds * 1e6, iters);
    if (rank == 0 && average_seconds > 0.0) {
        double bandwidth = static_cast<double>(bytes) / average_seconds / 1e9;
        LOG("Rank-average CPU time: %.3f us/iteration; TX bandwidth: %.6f GB/s",
            average_seconds * 1e6, bandwidth);
    } else if (rank == 0) {
        LOG("WARNING: CPU timing interval is 0; bandwidth unavailable");
    }

    LOG("Copying receive buffer from GPU to host");
    CUDACHECK(cudaMemcpy(h_recv.data(), d_recv, bytes,
                         cudaMemcpyDeviceToHost));

    LOG("Validating received data");
    bool ok = true;
    for (size_t i = 0; i < count; ++i) {
        if (h_recv[i] != float(recvPeer)) {
            LOG("Validation mismatch at index=%zu: got=%f expected=%f",
                i, h_recv[i], float(recvPeer));
            ok = false;
            break;
        }
    }
    LOG("Validation result: %s", ok ? "OK" : "NG");

    LOG("Releasing GPU buffers");
    if (use_nccl_alloc) {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 19, 0)
        NCCLCHECK(ncclMemFree(d_send));
        NCCLCHECK(ncclMemFree(d_recv));
#endif
    } else {
        CUDACHECK(cudaFree(d_send));
        CUDACHECK(cudaFree(d_recv));
    }
    LOG("Destroying CUDA stream");
    CUDACHECK(cudaStreamDestroy(stream));
    LOG("Destroying NCCL communicator");
    NCCLCHECK(ncclCommDestroy(comm));
    MPICHECK(MPI_Comm_free(&localComm));

    int localOk = ok ? 1 : 0;
    int globalOk = 0;
    LOG("Collecting validation results");
    MPICHECK(MPI_Allreduce(&localOk, &globalOk, 1, MPI_INT,
                           MPI_MIN, MPI_COMM_WORLD));
    LOG("Global validation result: %s", globalOk ? "OK" : "NG");

    LOG("Finalizing MPI");
    MPICHECK(MPI_Finalize());
    return globalOk ? 0 : 1;
}
