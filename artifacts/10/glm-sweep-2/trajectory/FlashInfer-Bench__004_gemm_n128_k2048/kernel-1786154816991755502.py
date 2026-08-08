import torch
import os

_CPP_SOURCE = r"""
#include <torch/extension.h>
#include <hipblaslt/hipblaslt.h>
#include <hip/hip_runtime.h>
#include <unordered_map>
#include <mutex>
#include <vector>

namespace {
struct GemmPlan {
    hipblasLtHandle_t handle = nullptr;
    hipblasLtMatmulPreference_t pref = nullptr;
    void* workspace = nullptr;
    size_t workspaceSize = 128 * 1024 * 1024;
    bool global_init = false;
};

struct PerShapePlan {
    hipblasLtMatmulDesc_t matmulDesc = nullptr;
    hipblasLtMatrixLayout_t Bdesc = nullptr;
    hipblasLtMatrixLayout_t Adesc = nullptr;
    hipblasLtMatrixLayout_t Cdesc = nullptr;
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristics;
    int best_idx = 0;
    bool tuned = false;
};

static GemmPlan g_plan;
static std::mutex g_mtx;
static std::unordered_map<int64_t, PerShapePlan> g_shape_plans;

void global_init() {
    if (g_plan.global_init) return;
    hipblasLtCreate(&g_plan.handle);
    void* ws = nullptr;
    if (hipMalloc(&ws, g_plan.workspaceSize) != hipSuccess) { ws = nullptr; }
    g_plan.workspace = ws;
    hipblasLtMatmulPreferenceCreate(&g_plan.pref);
    uint64_t wsz = g_plan.workspaceSize;
    hipblasLtMatmulPreferenceSetAttribute(
        g_plan.pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsz, sizeof(wsz));
    g_plan.global_init = true;
}

PerShapePlan& get_plan(int64_t M) {
    auto it = g_shape_plans.find(M);
    if (it != g_shape_plans.end()) return it->second;
    std::lock_guard<std::mutex> lk(g_mtx);
    global_init();
    it = g_shape_plans.find(M);
    if (it != g_shape_plans.end()) return it->second;

    PerShapePlan p;
    const int64_t K = 2048;
    const int64_t N = 128;
    // C_row = A_row @ B_row^T.  Using col-major interpretation of the same
    // row-major memory: B_row[N,K] ~ cm(K,N,ld=K) with opA=T;
    // A_row[M,K] ~ cm(K,M,ld=K) with opB=N; C_row[M,N] ~ cm(N,M,ld=N).
    hipblasLtMatrixLayoutCreate(&p.Bdesc, HIP_R_16F, K, N, K);
    hipblasLtMatrixLayoutCreate(&p.Adesc, HIP_R_16F, K, M, K);
    hipblasLtMatrixLayoutCreate(&p.Cdesc, HIP_R_16F, N, M, N);
    hipblasLtMatmulDescCreate(&p.matmulDesc, HIPBLAS_COMPUTE_32F, HIP_R_32F);
    hipblasOperation_t opA = HIPBLAS_OP_T;
    hipblasOperation_t opB = HIPBLAS_OP_N;
    hipblasLtMatmulDescSetAttribute(p.matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
    hipblasLtMatmulDescSetAttribute(p.matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));

    const int maxAlgos = 200;
    std::vector<hipblasLtMatmulHeuristicResult_t> heur(maxAlgos);
    int returnedCount = 0;
    hipblasLtMatmulAlgoGetHeuristic(g_plan.handle, p.matmulDesc,
                                    p.Bdesc, p.Adesc, p.Cdesc, p.Cdesc,
                                    g_plan.pref, maxAlgos, heur.data(), &returnedCount);
    p.heuristics.clear();
    for (int i = 0; i < returnedCount; i++) {
        if (heur[i].state == HIPBLAS_STATUS_SUCCESS) {
            p.heuristics.push_back(heur[i]);
        }
    }
    p.best_idx = 0;
    p.tuned = false;
    it = g_shape_plans.emplace(M, std::move(p)).first;
    return it->second;
}
}  // namespace

at::Tensor hipblas_gemm_run(at::Tensor A, at::Tensor B, int64_t stream_ptr) {
    int64_t M = A.size(0);
    PerShapePlan& p = get_plan(M);
    auto C = at::empty({M, 128}, A.options());
    hipStream_t stream = reinterpret_cast<hipStream_t>(stream_ptr);
    float alpha = 1.0f, beta = 0.0f;
    const hipblasLtMatmulAlgo_t* algo = nullptr;
    if (!p.heuristics.empty()) algo = &p.heuristics[p.best_idx].algo;
    hipblasLtMatmul(g_plan.handle, p.matmulDesc, &alpha,
                    B.data_ptr(), p.Bdesc, A.data_ptr(), p.Adesc,
                    &beta, C.data_ptr(), p.Cdesc, C.data_ptr(), p.Cdesc,
                    algo, g_plan.workspace, g_plan.workspaceSize, stream);
    return C;
}

int64_t hipblas_gemm_num_algos(int64_t M) {
    return (int64_t)get_plan(M).heuristics.size();
}

void hipblas_gemm_set_algo(int64_t M, int64_t idx) {
    PerShapePlan& p = get_plan(M);
    if (idx >= 0 && idx < (int64_t)p.heuristics.size()) {
        p.best_idx = (int)idx;
        p.tuned = true;
    }
}

int64_t hipblas_gemm_get_algo(int64_t M) {
    return (int64_t)get_plan(M).best_idx;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &hipblas_gemm_run, "GEMM");
    m.def("num_algos", &hipblas_gemm_num_algos, "num algos");
    m.def("set_algo", &hipblas_gemm_set_algo, "set algo");
    m.def("get_algo", &hipblas_gemm_get_algo, "get algo");
}
"""

_mod = None
_tuned = set()

# Pre-tuned best algorithm indices, derived from exhaustive sweep on MI350X.
# Keyed by M. algo 0 is the hipblaslt heuristic default; 1/3 are faster for
# specific skinny (N=128, K=2048) shapes.
_BEST_ALGO = {
    1: 3, 2: 0, 4: 0, 5: 1, 6: 0, 8: 0, 16: 1, 17: 1, 25: 1, 32: 1,
    34: 1, 63: 1, 64: 1, 93: 1, 128: 1, 172: 1, 289: 1, 492: 0, 952: 1,
    8828: 0, 11006: 0, 12251: 0, 14915: 0, 16294: 0,
}


def _get_mod():
    global _mod
    if _mod is not None:
        return _mod
    from torch.utils.cpp_extension import load
    src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_hipblas_gemm_src.cpp")
    with open(src_path, "w") as f:
        f.write(_CPP_SOURCE)
    _mod = load(
        name="hipblas_gemm_kernel",
        sources=[src_path],
        extra_include_paths=["/opt/rocm-7.2.0/include"],
        extra_ldflags=[
            "-L/opt/rocm-7.2.0/lib", "-lhipblaslt",
            "-Wl,-rpath,/opt/rocm-7.2.0/lib",
        ],
        verbose=False,
    )
    return _mod


def run(A, B):
    mod = _get_mod()
    M = A.shape[0]
    if M not in _tuned:
        algo = _BEST_ALGO.get(M, 0)
        n = mod.num_algos(M)
        if algo >= n:
            algo = 0
        mod.set_algo(M, algo)
        _tuned.add(M)
    stream = torch.cuda.current_stream().cuda_stream
    return mod.run(A, B, stream)
