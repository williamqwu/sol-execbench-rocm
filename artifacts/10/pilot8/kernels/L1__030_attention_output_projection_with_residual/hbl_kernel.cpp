// hipBLASLt fused GEMM + residual (beta=1)
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <vector>
#include <map>
#include <mutex>

#define CHECK(x) TORCH_CHECK((x) == HIPBLAS_STATUS_SUCCESS, "hipblaslt fail at " #x)

static hipblasLtHandle_t g_handle = nullptr;
static void* g_ws = nullptr;
static size_t g_ws_size = 128 * 1024 * 1024;

struct Key {
  int64_t M, N, K, algo_idx;
  bool operator<(const Key& o) const {
    return std::tie(M, N, K, algo_idx) < std::tie(o.M, o.N, o.K, o.algo_idx);
  }
};

struct Plan {
  hipblasLtMatmulDesc_t desc;
  hipblasLtMatrixLayout_t la, lb, lc, ld;
  hipblasLtMatmulAlgo_t algo;
  size_t ws;
  bool valid;
};

static std::map<Key, Plan> g_plans;
static std::map<Key, int> g_nalgo;

static void init() {
  if (!g_handle) {
    CHECK(hipblasLtCreate(&g_handle));
    TORCH_CHECK(hipMalloc(&g_ws, g_ws_size) == hipSuccess, "ws alloc");
  }
}

// Build plan for logical row-major C[M,N] = A[M,K] @ W[N,K]^T + C
// col-major mapping: m=N, n=M, k=K, opA=T (W, ld=K), opB=N (A, ld=K), ldc=ldd=N
static Plan* get_plan(int64_t M, int64_t N, int64_t K, int algo_idx, int* n_avail) {
  Key key{M, N, K, algo_idx};
  auto it = g_plans.find(key);
  if (it != g_plans.end()) {
    if (n_avail) {
      auto n = g_nalgo.find(Key{M, N, K, -1});
      *n_avail = (n != g_nalgo.end()) ? n->second : 0;
    }
    return it->second.valid ? &it->second : nullptr;
  }

  Plan p{};
  p.valid = false;
  CHECK(hipblasLtMatmulDescCreate(&p.desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
  hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;
  CHECK(hipblasLtMatmulDescSetAttribute(p.desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)));
  CHECK(hipblasLtMatmulDescSetAttribute(p.desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));
  CHECK(hipblasLtMatrixLayoutCreate(&p.la, HIP_R_16BF, K, N, K));
  CHECK(hipblasLtMatrixLayoutCreate(&p.lb, HIP_R_16BF, K, M, K));
  CHECK(hipblasLtMatrixLayoutCreate(&p.lc, HIP_R_16BF, N, M, N));
  CHECK(hipblasLtMatrixLayoutCreate(&p.ld, HIP_R_16BF, N, M, N));

  hipblasLtMatmulPreference_t pref;
  CHECK(hipblasLtMatmulPreferenceCreate(&pref));
  CHECK(hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &g_ws_size, sizeof(g_ws_size)));
  const int kMax = 64;
  std::vector<hipblasLtMatmulHeuristicResult_t> res(kMax);
  int ret = 0;
  hipblasLtMatmulAlgoGetHeuristic(g_handle, p.desc, p.la, p.lb, p.lc, p.ld, pref, kMax,
                                  res.data(), &ret);
  hipblasLtMatmulPreferenceDestroy(pref);
  g_nalgo[Key{M, N, K, -1}] = ret;
  if (n_avail) *n_avail = ret;

  if (algo_idx < ret) {
    p.algo = res[algo_idx].algo;
    p.ws = res[algo_idx].workspaceSize;
    p.valid = true;
  }
  g_plans[key] = p;
  return p.valid ? &g_plans[key] : nullptr;
}

int64_t num_algos(int64_t M, int64_t N, int64_t K) {
  init();
  int n = 0;
  get_plan(M, N, K, 0, &n);
  return n;
}

torch::Tensor fused(torch::Tensor a, torch::Tensor r, torch::Tensor w, int64_t algo_idx) {
  init();
  int64_t M = a.size(0), K = a.size(1), N = w.size(0);
  auto out = torch::empty({M, N}, a.options());
  Plan* p = get_plan(M, N, K, (int)algo_idx, nullptr);
  TORCH_CHECK(p != nullptr, "no algo ", algo_idx);
  float alpha = 1.0f, beta = 1.0f;
  auto stream = at::hip::getCurrentHIPStream().stream();
  auto st = hipblasLtMatmul(g_handle, p->desc, &alpha, w.data_ptr(), p->la, a.data_ptr(), p->lb,
                            &beta, r.data_ptr(), p->lc, out.data_ptr(), p->ld, &p->algo, g_ws,
                            g_ws_size, stream);
  TORCH_CHECK(st == HIPBLAS_STATUS_SUCCESS, "matmul failed ", (int)st);
  return out;
}

void fused_out(torch::Tensor a, torch::Tensor r, torch::Tensor w, torch::Tensor out,
               int64_t algo_idx) {
  init();
  int64_t M = a.size(0), K = a.size(1), N = w.size(0);
  Plan* p = get_plan(M, N, K, (int)algo_idx, nullptr);
  TORCH_CHECK(p != nullptr, "no algo ", algo_idx);
  float alpha = 1.0f, beta = 1.0f;
  auto stream = at::hip::getCurrentHIPStream().stream();
  auto st = hipblasLtMatmul(g_handle, p->desc, &alpha, w.data_ptr(), p->la, a.data_ptr(), p->lb,
                            &beta, r.data_ptr(), p->lc, out.data_ptr(), p->ld, &p->algo, g_ws,
                            g_ws_size, stream);
  TORCH_CHECK(st == HIPBLAS_STATUS_SUCCESS, "matmul failed ", (int)st);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused", &fused);
  m.def("fused_out", &fused_out);
  m.def("num_algos", &num_algos);
}
