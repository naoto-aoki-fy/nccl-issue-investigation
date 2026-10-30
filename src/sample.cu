#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>

#include <getopt.h>
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
static void wait_for_transfer(cudaStream_t stream, ncclComm_t comm,
                              const std::string& mode, cudaEvent_t stop_event = nullptr) {
    if (mode == "stream") {
        CUDACHECK(cudaStreamSynchronize(stream));
        return;
    }
    if (mode == "event") {
        CUDACHECK(cudaEventSynchronize(stop_event));
        return;
    }
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
        "  --comm-init MODE       config (default) or rank\n"
        "  --max-p2p-peers VALUE  default or positive integer (default: 2; NCCL 2.30+)\n"
        "  --nvlink-centric-sched VALUE  default, 0 or 1 (default: 1; NCCL 2.28+)\n"
        "  --graph-usage-mode VALUE  default, 0, 1 or 2 (default: 0; NCCL 2.29+)\n"
        "  --stream MODE          nonblocking (default) or blocking\n"
        "  --wait MODE            poll (default), stream or event\n"
        "  --warmup-wait MODE     poll (default) or stream\n"
        "  --pre-log POSITION     before (default) or after the measurement barrier\n"
        "  --timing MODE          cpu (default), event or both\n"
        "  --rank-report MODE     both (default), average or per-rank\n"
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
    std::string comm_init = "config", max_peers = "2", nvlink_sched = "1", graph_mode = "0";
    std::string stream_mode = "nonblocking", wait_mode = "poll", warmup_wait = "poll", pre_log = "before";
    std::string timing_mode = "cpu", rank_report = "both";
    bool max_peers_explicit = false, nvlink_explicit = false, graph_explicit = false;
    std::string parse_error;

    opterr = 0;
    enum { OPT_COMM_INIT = 256, OPT_MAX_PEERS, OPT_NVLINK, OPT_GRAPH,
           OPT_STREAM, OPT_WAIT, OPT_WARMUP_WAIT, OPT_PRE_LOG, OPT_TIMING, OPT_RANK_REPORT };
    const option long_options[] = {
        {"comm-init", required_argument, nullptr, OPT_COMM_INIT},
        {"max-p2p-peers", required_argument, nullptr, OPT_MAX_PEERS},
        {"nvlink-centric-sched", required_argument, nullptr, OPT_NVLINK},
        {"graph-usage-mode", required_argument, nullptr, OPT_GRAPH},
        {"stream", required_argument, nullptr, OPT_STREAM},
        {"wait", required_argument, nullptr, OPT_WAIT},
        {"warmup-wait", required_argument, nullptr, OPT_WARMUP_WAIT},
        {"pre-log", required_argument, nullptr, OPT_PRE_LOG},
        {"timing", required_argument, nullptr, OPT_TIMING},
        {"rank-report", required_argument, nullptr, OPT_RANK_REPORT},
        {nullptr, 0, nullptr, 0}
    };
    int opt;
    while ((opt = getopt_long(argc, argv, "s:n:w:W:A:h", long_options, nullptr)) != -1) {
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
        case OPT_COMM_INIT:
            comm_init = optarg;
            if (comm_init != "config" && comm_init != "rank") parse_error = "invalid --comm-init (config or rank)";
            break;
        case OPT_MAX_PEERS: {
            max_peers = optarg;
            max_peers_explicit = true;
            int peers = 0;
            if (max_peers != "default" && (!parse_iteration_count(optarg, &peers) || peers < 1 || peers > nranks))
                parse_error = "invalid --max-p2p-peers (default or 1..nranks)";
            break;
        }
        case OPT_NVLINK:
            nvlink_sched = optarg;
            nvlink_explicit = true;
            if (nvlink_sched != "default" && nvlink_sched != "0" && nvlink_sched != "1")
                parse_error = "invalid --nvlink-centric-sched (default, 0 or 1)";
            break;
        case OPT_GRAPH:
            graph_mode = optarg;
            graph_explicit = true;
            if (graph_mode != "default" && graph_mode != "0" && graph_mode != "1" && graph_mode != "2")
                parse_error = "invalid --graph-usage-mode (default, 0, 1 or 2)";
            break;
        case OPT_STREAM:
            stream_mode = optarg;
            if (stream_mode != "nonblocking" && stream_mode != "blocking") parse_error = "invalid --stream (nonblocking or blocking)";
            break;
        case OPT_WAIT:
            wait_mode = optarg;
            if (wait_mode != "poll" && wait_mode != "stream" && wait_mode != "event") parse_error = "invalid --wait (poll, stream or event)";
            break;
        case OPT_WARMUP_WAIT:
            warmup_wait = optarg;
            if (warmup_wait != "poll" && warmup_wait != "stream") parse_error = "invalid --warmup-wait (poll or stream)";
            break;
        case OPT_PRE_LOG:
            pre_log = optarg;
            if (pre_log != "before" && pre_log != "after") parse_error = "invalid --pre-log (before or after)";
            break;
        case OPT_TIMING:
            timing_mode = optarg;
            if (timing_mode != "cpu" && timing_mode != "event" && timing_mode != "both") parse_error = "invalid --timing (cpu, event or both)";
            break;
        case OPT_RANK_REPORT:
            rank_report = optarg;
            if (rank_report != "both" && rank_report != "average" && rank_report != "per-rank") parse_error = "invalid --rank-report (both, average or per-rank)";
            break;
        default:
            parse_error = "unknown option or missing option argument";
            break;
        }
    }
    if (optind < argc) parse_error = "unexpected positional arguments";
    if (comm_init == "rank" && (max_peers_explicit || nvlink_explicit || graph_explicit))
        parse_error = "--comm-init rank cannot be combined with explicit communicator config options";
#if NCCL_VERSION_CODE < NCCL_VERSION(2, 14, 0)
    if (comm_init == "config") comm_init = "rank";
#endif
#if NCCL_VERSION_CODE < NCCL_VERSION(2, 28, 0)
    if (nvlink_explicit && nvlink_sched != "default") parse_error = "--nvlink-centric-sched requires NCCL 2.28+ headers";
#endif
#if NCCL_VERSION_CODE < NCCL_VERSION(2, 29, 0)
    if (graph_explicit && graph_mode != "default") parse_error = "--graph-usage-mode requires NCCL 2.29+ headers";
#endif
#if NCCL_VERSION_CODE < NCCL_VERSION(2, 30, 0)
    if (max_peers_explicit && max_peers != "default") parse_error = "--max-p2p-peers requires NCCL 2.30+ headers";
#endif

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
    LOG("Measurement configuration: iterations=%d timing=%s rank-report=%s wait=%s warmup-wait=%s pre-log=%s stream=%s",
        iters, timing_mode.c_str(), rank_report.c_str(), wait_mode.c_str(),
        warmup_wait.c_str(), pre_log.c_str(), stream_mode.c_str());
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
    if (comm_init == "config") {
        ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
        if (nvlink_sched != "default") config.nvlinkCentricSched = std::atoi(nvlink_sched.c_str());
#endif
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 0)
        if (graph_mode != "default") config.graphUsageMode = std::atoi(graph_mode.c_str());
#endif
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 0)
        if (max_peers != "default") config.maxP2pPeers = std::atoi(max_peers.c_str());
#endif
        NCCLCHECK(ncclCommInitRankConfig(&comm, nranks, id, rank, &config));
    } else
#endif
    {
        NCCLCHECK(ncclCommInitRank(&comm, nranks, id, rank));
    }
    LOG("NCCL communicator initialized");
    if (comm_init == "rank") {
        LOG("Communicator configuration: rank initialization (library defaults)");
    } else {
        LOG("Communicator requested configuration: nvlinkCentricSched=%s graphUsageMode=%s maxP2pPeers=%s (unsupported header versions use library defaults; environment may override)",
            nvlink_sched.c_str(), graph_mode.c_str(), max_peers.c_str());
    }

    LOG("Creating %s CUDA stream", stream_mode.c_str());
    cudaStream_t stream;
    CUDACHECK(cudaStreamCreateWithFlags(&stream, stream_mode == "nonblocking" ? cudaStreamNonBlocking : cudaStreamDefault));

    // Only create and record events when requested: event instrumentation can
    // change the measured CPU time, so the default path remains unchanged.
    const bool measure_event = timing_mode != "cpu";
    const bool need_stop_event = measure_event || wait_mode == "event";
    cudaEvent_t start_event = nullptr, stop_event = nullptr;
    if (measure_event) CUDACHECK(cudaEventCreate(&start_event));
    if (need_stop_event) CUDACHECK(cudaEventCreate(&stop_event));

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
    wait_for_transfer(stream, comm, warmup_wait);
    if (pre_log == "before") {
        LOG("Starting %d consecutive NCCL transfers: %.2f MiB TX + %.2f MiB RX each",
            iters, static_cast<double>(bytes) / MiB, static_cast<double>(bytes) / MiB);
    }

    // Log before this barrier to avoid rank skew from progress output.
    // Match sendrecv_perf -m 1 -z 0 -G 0: one group per iteration,
    // no intermediate synchronization, and one completion wait at the end.
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (pre_log == "after") {
        LOG("Starting %d consecutive NCCL transfers: %.2f MiB TX + %.2f MiB RX each",
            iters, static_cast<double>(bytes) / MiB, static_cast<double>(bytes) / MiB);
    }
    auto measurement_start = std::chrono::steady_clock::now();
    if (measure_event) CUDACHECK(cudaEventRecord(start_event, stream));
    for (int i = 0; i < iters; ++i) {
        NCCLCHECK(ncclGroupStart());
        NCCLCHECK(ncclSend(d_send, count, ncclFloat, sendPeer, comm, stream));
        NCCLCHECK(ncclRecv(d_recv, count, ncclFloat, recvPeer, comm, stream));
        NCCLCHECK(ncclGroupEnd());
    }
    if (need_stop_event) CUDACHECK(cudaEventRecord(stop_event, stream));
    wait_for_transfer(stream, comm, wait_mode, stop_event);
    double local_cpu_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - measurement_start).count() / iters;
    double local_event_seconds = 0.0;
    if (measure_event) {
        float event_ms = 0.0f;
        CUDACHECK(cudaEventElapsedTime(&event_ms, start_event, stop_event));
        local_event_seconds = static_cast<double>(event_ms) / 1000.0 / iters;
    }

    // Match sendrecv_perf -a 1: average the per-iteration times across ranks,
    // then calculate bandwidth using one sent message (not TX+RX).
    double local_times[2] = {local_cpu_seconds, local_event_seconds};
    double sum_times[2] = {0.0, 0.0};
    MPICHECK(MPI_Allreduce(local_times, sum_times, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
    for (int i = 0; i < 2; ++i) {
        if ((i == 0 && timing_mode == "event") || (i == 1 && !measure_event)) continue;
        const char* label = i == 0 ? "CPU" : "CUDA event";
        if (rank_report != "average") {
            LOG("Local %s time: %.3f us/iteration (%d iterations)", label, local_times[i] * 1e6, iters);
            if (local_times[i] > 0.0)
                LOG("Local %s bandwidth: TX=%.6f GB/s, TX+RX=%.6f GB/s",
                    label, static_cast<double>(bytes) / local_times[i] / 1e9,
                    2.0 * static_cast<double>(bytes) / local_times[i] / 1e9);
        }
        double average_seconds = sum_times[i] / nranks;
        if (rank == 0 && rank_report != "per-rank") {
            if (average_seconds > 0.0) {
                double bandwidth = static_cast<double>(bytes) / average_seconds / 1e9;
                LOG("Rank-average %s time: %.3f us/iteration; TX bandwidth: %.6f GB/s",
                    label, average_seconds * 1e6, bandwidth);
            } else {
                LOG("WARNING: %s timing interval is 0; bandwidth unavailable", label);
            }
        }
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

    if (measure_event) CUDACHECK(cudaEventDestroy(start_event));
    if (need_stop_event) CUDACHECK(cudaEventDestroy(stop_event));
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
