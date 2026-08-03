// SPDX-License-Identifier: Apache-2.0
//
// Task 04 — rocprofiler-sdk activity shim.
//
// This is the ONLY vendor-specific piece of the AMD timing path. Every
// decision about which activities belong to which timed iteration -- window
// bisection, sequence selection, span attribution -- already lives in
// src/solexbench_rocm/activity/, is behaviour-preserving against upstream's
// CUPTI logic, and is covered by mutation-tested CPU tests. This file only
// supplies records.
//
// Contract (reference/contracts/rocprof_shim.md):
//
//     start()      -> None    configure + start buffered tracing
//     stop()       -> None    stop and flush
//     drain()      -> list    activity tuples
//     timestamp()  -> int     host timestamp in the RECORD clock domain
//
// tuple = (kind, name, start_ns, end_ns, correlation_id, copy_kind, nbytes, value)
//
// The five traps from the contract, and what is done about each:
//
//   #1 CLOCK DOMAIN. `timestamp()` calls rocprofiler_get_timestamp(), NOT
//      clock_gettime(CLOCK_MONOTONIC). Record stamps come from the HSA clock;
//      mixing domains does not raise, it just makes bisection select the wrong
//      activities, and the resulting distribution looks plausible. This is the
//      single most expensive mistake available in this file.
//
//   #2 NAME RESOLUTION. Dispatch records carry a kernel_id, not a string. Ids
//      are resolved through the code-object callback and cached; the cache is
//      keyed by id and entries are dropped when the owning code object
//      unloads, because ids are only valid for the loaded code object.
//
//   #3 FLUSH ORDERING. drain() flushes the buffer first and is callable after
//      stop(). No assumption is made about arrival order -- the pure layer
//      sorts defensively, and this file does not re-sort.
//
//   #4 DISPATCH-LEVEL, NOT API-LEVEL. Only KERNEL_DISPATCH and MEMORY_COPY
//      buffered categories are traced. Tracing HIP_RUNTIME_API instead would
//      reintroduce host launch overhead -- the exact cost this methodology
//      exists to exclude -- while appearing to work.
//
//   #5 MEMSET. rocprofiler-sdk has no distinct buffered "memset" category; the
//      harness's own 512 MB LLC flush arrives as a DEVICE_TO_DEVICE copy. It
//      is emitted as MEMCPY with its true copy_kind and byte count, so
//      identity() stays discriminating. See the note in the Python wrapper:
//      the flush is filtered by identity, and a mislabelled kind would break
//      that filter rather than silently pass.

// HIP first: rocprofiler-sdk's umbrella header pulls in its RCCL API argument
// structs, which name `hipStream_t` without including HIP themselves. Without
// this the build fails on a dozen "does not name a type" errors inside a
// header this file never uses.
#define __HIP_PLATFORM_AMD__ 1
#include <hip/hip_runtime.h>

#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

namespace {

struct Activity {
  std::string kind;
  std::string name;
  uint64_t start_ns = 0;
  uint64_t end_ns = 0;
  uint64_t correlation_id = 0;
  int64_t copy_kind = 0;
  uint64_t nbytes = 0;
  uint64_t value = 0;
};

// Everything the callbacks touch. Guarded because rocprofiler delivers buffer
// callbacks on its own thread.
std::mutex g_mutex;
std::vector<Activity> g_records;
std::unordered_map<uint64_t, std::string> g_kernel_names;

rocprofiler_context_id_t g_context = {0};
rocprofiler_buffer_id_t g_buffer = {0};
bool g_configured = false;
std::string g_config_error;

// -- #2 name resolution -----------------------------------------------------
void code_object_callback(rocprofiler_callback_tracing_record_t record,
                          rocprofiler_user_data_t*, void*) {
  if (record.kind != ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT) return;
  if (record.operation !=
      ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER) {
    return;
  }
  auto* data = static_cast<
      rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t*>(
      record.payload);
  if (data == nullptr) return;

  std::lock_guard<std::mutex> lock(g_mutex);
  if (record.phase == ROCPROFILER_CALLBACK_PHASE_LOAD) {
    g_kernel_names[data->kernel_id] =
        data->kernel_name ? std::string(data->kernel_name) : std::string();
  } else if (record.phase == ROCPROFILER_CALLBACK_PHASE_UNLOAD) {
    // Ids are only valid while the code object is loaded. Keeping the entry
    // would hand a later dispatch the previous module's name -- a wrong label
    // on a real measurement, which is worse than no label.
    g_kernel_names.erase(data->kernel_id);
  }
}

// -- #3/#4 buffered records -------------------------------------------------
void buffer_callback(rocprofiler_context_id_t, rocprofiler_buffer_id_t,
                     rocprofiler_record_header_t** headers,
                     size_t num_headers, void*, uint64_t) {
  std::lock_guard<std::mutex> lock(g_mutex);
  for (size_t i = 0; i < num_headers; ++i) {
    auto* header = headers[i];
    if (header == nullptr ||
        header->category != ROCPROFILER_BUFFER_CATEGORY_TRACING) {
      continue;
    }

    if (header->kind == ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH) {
      auto* rec =
          static_cast<rocprofiler_buffer_tracing_kernel_dispatch_record_t*>(
              header->payload);
      Activity a;
      a.kind = "KERNEL";
      auto it = g_kernel_names.find(rec->dispatch_info.kernel_id);
      // A kernel whose name never arrived is labelled by id rather than left
      // blank: the pure layer keys identity on the name, and two differently
      // named kernels collapsing to "" would corrupt sequence selection.
      a.name = (it != g_kernel_names.end())
                   ? it->second
                   : ("kernel_id_" +
                      std::to_string(rec->dispatch_info.kernel_id));
      a.start_ns = rec->start_timestamp;
      a.end_ns = rec->end_timestamp;
      a.correlation_id = rec->correlation_id.internal;
      g_records.push_back(std::move(a));

    } else if (header->kind == ROCPROFILER_BUFFER_TRACING_MEMORY_COPY) {
      auto* rec =
          static_cast<rocprofiler_buffer_tracing_memory_copy_record_t*>(
              header->payload);
      Activity a;
      a.kind = "MEMCPY";
      a.name = "MemoryCopy";
      a.start_ns = rec->start_timestamp;
      a.end_ns = rec->end_timestamp;
      a.correlation_id = rec->correlation_id.internal;
      a.copy_kind = static_cast<int64_t>(rec->operation);
      a.nbytes = rec->bytes;
      g_records.push_back(std::move(a));
    }
  }
}

int tool_init(rocprofiler_client_finalize_t, void*) {
  if (rocprofiler_create_context(&g_context) != ROCPROFILER_STATUS_SUCCESS) {
    g_config_error = "rocprofiler_create_context failed";
    return -1;
  }

  if (rocprofiler_configure_callback_tracing_service(
          g_context, ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT, nullptr, 0,
          code_object_callback, nullptr) != ROCPROFILER_STATUS_SUCCESS) {
    g_config_error = "code-object callback service failed";
    return -1;
  }

  constexpr size_t kBufferBytes = 16 * 1024 * 1024;
  constexpr size_t kWatermark = 8 * 1024 * 1024;
  if (rocprofiler_create_buffer(g_context, kBufferBytes, kWatermark,
                                ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                buffer_callback, nullptr,
                                &g_buffer) != ROCPROFILER_STATUS_SUCCESS) {
    // LOSSLESS on purpose: a dropped record silently removes work from an
    // iteration's span and makes a kernel look faster than it is.
    g_config_error = "rocprofiler_create_buffer failed";
    return -1;
  }

  for (auto kind : {ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
                    ROCPROFILER_BUFFER_TRACING_MEMORY_COPY}) {
    if (rocprofiler_configure_buffer_tracing_service(
            g_context, kind, nullptr, 0, g_buffer) !=
        ROCPROFILER_STATUS_SUCCESS) {
      g_config_error = "buffer tracing service failed";
      return -1;
    }
  }

  int valid = 0;
  if (rocprofiler_context_is_valid(g_context, &valid) !=
          ROCPROFILER_STATUS_SUCCESS ||
      valid == 0) {
    g_config_error = "context invalid after configuration";
    return -1;
  }

  g_configured = true;
  return 0;
}

}  // namespace

extern "C" rocprofiler_tool_configure_result_t* rocprofiler_configure(
    uint32_t /*version*/, const char* /*runtime_version*/, uint32_t /*priority*/,
    rocprofiler_client_id_t* client_id) {
  client_id->name = "solexbench-rocprof-shim";
  static auto cfg = rocprofiler_tool_configure_result_t{
      sizeof(rocprofiler_tool_configure_result_t), &tool_init, nullptr,
      nullptr};
  return &cfg;
}

namespace {

void ensure_configured() {
  if (g_configured) return;
  // Late registration. rocprofiler normally scans for rocprofiler_configure at
  // library load; a Python extension imported after the HIP runtime has begun
  // initializing has missed that scan, and force_configure is the documented
  // way back in. It fails once configuration is locked, which is why the
  // Python wrapper imports this module before touching torch.cuda.
  auto status = rocprofiler_force_configure(&rocprofiler_configure);
  if (status != ROCPROFILER_STATUS_SUCCESS && !g_configured) {
    throw std::runtime_error(
        std::string("rocprofiler configuration failed (") +
        rocprofiler_get_status_string(status) + ")" +
        (g_config_error.empty() ? "" : ": " + g_config_error) +
        ". Import _rocprof_shim BEFORE any HIP work: once the runtime has "
        "initialized, configuration is locked.");
  }
}

void start() {
  ensure_configured();
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_records.clear();
  }
  auto status = rocprofiler_start_context(g_context);
  if (status != ROCPROFILER_STATUS_SUCCESS) {
    throw std::runtime_error(std::string("rocprofiler_start_context: ") +
                             rocprofiler_get_status_string(status));
  }
}

void stop() {
  auto status = rocprofiler_stop_context(g_context);
  if (status != ROCPROFILER_STATUS_SUCCESS) {
    throw std::runtime_error(std::string("rocprofiler_stop_context: ") +
                             rocprofiler_get_status_string(status));
  }
  rocprofiler_flush_buffer(g_buffer);
}

py::list drain() {
  // Flush again: stop() may have been called while records were still in
  // flight, and a record that arrives after drain() belongs to an iteration
  // that has already been attributed.
  if (g_configured) rocprofiler_flush_buffer(g_buffer);

  std::vector<Activity> taken;
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    taken.swap(g_records);
  }
  py::list out;
  for (const auto& a : taken) {
    out.append(py::make_tuple(a.kind, a.name, a.start_ns, a.end_ns,
                              a.correlation_id, a.copy_kind, a.nbytes,
                              a.value));
  }
  return out;
}

uint64_t timestamp() {
  // TRAP #1. This must be rocprofiler's own clock, not CLOCK_MONOTONIC.
  rocprofiler_timestamp_t ts = 0;
  auto status = rocprofiler_get_timestamp(&ts);
  if (status != ROCPROFILER_STATUS_SUCCESS) {
    throw std::runtime_error(std::string("rocprofiler_get_timestamp: ") +
                             rocprofiler_get_status_string(status));
  }
  return static_cast<uint64_t>(ts);
}

}  // namespace

PYBIND11_MODULE(_rocprof_shim, m) {
  m.doc() = "rocprofiler-sdk activity source for SOL-ExecBench-AMD";
  m.def("configure", &ensure_configured,
        "Register with rocprofiler. MUST be called before the HIP runtime "
        "initializes: rocprofiler locks its configuration once a runtime is "
        "up, and a session configured too late produces zero records rather "
        "than an error.");
  m.def("start", &start, "Configure and start buffered tracing");
  m.def("stop", &stop, "Stop tracing and flush");
  m.def("drain", &drain, "Return recorded activities as tuples");
  m.def("timestamp", &timestamp,
        "Host timestamp in the same clock domain as record timestamps");
  m.def("is_configured", []() { return g_configured; });
}
