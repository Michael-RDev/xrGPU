#include "devices/registry.h"
#include "scene.h"
#include "trace.h"
#include <dlfcn.h>
#include "runtime/native/dispatch.h"
#include "runtime/integration/xrgpu_kernel_package.h"
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/platform/runtime.h>
#include <jni.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/system_properties.h>
#include <vector>
#include <vulkan/vulkan.h>

namespace {
using namespace xrgpu;
using executorch::extension::make_tensor_ptr;
using executorch::extension::Module;
using executorch::runtime::EValue;

std::string string(JNIEnv *env, jstring input) {
  if (!input)
    throw std::runtime_error("null string");
  const char *chars = env->GetStringUTFChars(input, nullptr);
  if (!chars)
    throw std::runtime_error("string allocation failed");
  std::string result(chars);
  env->ReleaseStringUTFChars(input, chars);
  return result;
}
void failure(JNIEnv *env, const std::exception &e) {
  env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), e.what());
}
Json capabilities() {
  char model[PROP_VALUE_MAX]{};
  __system_property_get("ro.product.model", model);
  Json result = {{"vulkan", false},
                 {"gpu_timestamps", false},
                 {"candidate_capable", false},
                 {"device_model", model},
                 {"device_profile", "stock"},
                 {"kernel_set", "stock"},
                 {"candidate_variants", Json::array()},
                 {"variants", Json::array()},
                 {"kernel_revision", XRGPU_KERNEL_REVISION},
                 {"runtime_revision", XRGPU_ET_REVISION}};
  void *lib = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
  if (!lib) {
    result["reason"] = "libvulkan.so unavailable";
    return result;
  }
  auto proc = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
      dlsym(lib, "vkGetInstanceProcAddr"));
  if (!proc) {
    result["reason"] = "Vulkan entry point unavailable";
    dlclose(lib);
    return result;
  }
  auto create = reinterpret_cast<PFN_vkCreateInstance>(
      proc(VK_NULL_HANDLE, "vkCreateInstance"));
  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "xrGPU diagnostics";
  app.apiVersion = VK_API_VERSION_1_1;
  VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  info.pApplicationInfo = &app;
  VkInstance instance = VK_NULL_HANDLE;
  if (create(&info, nullptr, &instance) != VK_SUCCESS) {
    result["reason"] = "Vulkan 1.1 instance creation failed";
    dlclose(lib);
    return result;
  }
  auto enumerate = reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(
      proc(instance, "vkEnumeratePhysicalDevices"));
  auto properties = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(
      proc(instance, "vkGetPhysicalDeviceProperties"));
  auto queues = reinterpret_cast<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
      proc(instance, "vkGetPhysicalDeviceQueueFamilyProperties"));
  auto destroy = reinterpret_cast<PFN_vkDestroyInstance>(
      proc(instance, "vkDestroyInstance"));
  uint32_t count = 0;
  enumerate(instance, &count, nullptr);
  std::vector<VkPhysicalDevice> devices(count);
  enumerate(instance, &count, devices.data());
  if (count) {
    VkPhysicalDeviceProperties p{};
    properties(devices[0], &p);
    uint32_t nq = 0;
    queues(devices[0], &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qp(nq);
    queues(devices[0], &nq, qp.data());
    bool timestamps = false;
    for (auto &q : qp)
      if (q.queueFlags & VK_QUEUE_COMPUTE_BIT) {
        timestamps = q.timestampValidBits > 0;
        break;
      }
    const auto *profile = find_device_profile(
        {model, p.vendorID, p.limits.maxComputeWorkGroupInvocations,
         p.limits.maxComputeWorkGroupSize[0],
         p.limits.maxComputeWorkGroupSize[1], p.deviceID, p.apiVersion});
    if (profile && profile->linear) {
      result["device_profile"] = profile->id;
      result["kernel_set"] = profile->linear->id;
      for (size_t i = 0; i < profile->linear->variant_count; ++i) {
        const auto &v = profile->linear->variants[i];
        result["candidate_variants"].push_back(v.id);
        result["variants"].push_back({{"id", v.id}, {"label", v.label}});
      }
    }
    result.update(
        {{"vulkan", true},
         {"gpu", p.deviceName},
         {"vendor_id", p.vendorID},
         {"device_id", p.deviceID},
         {"driver", p.driverVersion},
         {"api_version", p.apiVersion},
         {"gpu_timestamps", timestamps && p.limits.timestampPeriod > 0},
         {"timestamp_period_ns", p.limits.timestampPeriod},
         {"max_workgroup_invocations", p.limits.maxComputeWorkGroupInvocations},
         {"candidate_capable",
          profile && profile->linear && profile->linear->variant_count > 0}});
  }
  destroy(instance, nullptr);
  dlclose(lib);
  return result;
}

struct Session {
  Trace *trace;
  std::unique_ptr<Module> module;
  int variant;
  const DeviceProfile *profile;
  uint64_t nodes;
  xrgpu_package::Context selection;
  double last_ms = 0, load_ms;
  explicit Session(const std::string &path, int requested)
      : variant(requested) {
    auto caps = capabilities();
    profile = profile_by_id(caps["device_profile"].get<std::string>());
    if (requested &&
        (!profile || !profile->linear || !profile->linear->variant(requested)))
      throw std::runtime_error("Candidate is unavailable for this device "
                               "profile; use stock execution");
    executorch::runtime::runtime_init();
    // The demo explicitly requests candidates; stock remains an untouched
    // baseline. Qualification and automatic production selection are separate.
    selection.require_repairs = false;
    if (requested)
      selection.experiment = "linear:" + std::string(profile->linear->variant(requested)->label);
    xrgpu_package::Scope selection_scope(selection);
    xrgpu_package::selections = 0;
    gpu_timestamps_supported = caps["gpu_timestamps"].get<bool>();
    auto t = std::make_unique<Trace>();
    trace = t.get();
    const double begin = milliseconds();
    module =
        std::make_unique<Module>(path, Module::LoadMode::Mmap, std::move(t));
    auto error = module->load_forward();
    if (error != executorch::runtime::Error::Ok)
      throw std::runtime_error("ExecuTorch load failed, code " +
                               std::to_string(static_cast<int>(error)));
    load_ms = milliseconds() - begin;
    nodes = xrgpu_package::selections;
  }
  std::vector<float> run(const std::vector<EValue> &input) {
    xrgpu_package::Scope selection_scope(selection);
    trace->reset();
    const double start = milliseconds();
    auto outputs = module->forward(input);
    last_ms = milliseconds() - start;
    if (!outputs.ok())
      throw std::runtime_error(
          "ExecuTorch forward failed, code " +
          std::to_string(static_cast<int>(outputs.error())));
    if (outputs->empty() || !outputs->at(0).isTensor())
      throw std::runtime_error("Missing output tensor");
    const auto tensor = outputs->at(0).toTensor();
    if (tensor.scalar_type() != executorch::aten::ScalarType::Float)
      throw std::runtime_error("Expected FP32 output");
    auto data = tensor.const_data_ptr<float>();
    return std::vector<float>(data, data + tensor.numel());
  }
};
Session &session(jlong handle) {
  if (!handle)
    throw std::runtime_error("Closed session");
  return *reinterpret_cast<Session *>(handle);
}
jfloatArray array(JNIEnv *env, const std::vector<float> &output) {
  auto result = env->NewFloatArray(static_cast<jsize>(output.size()));
  if (!result)
    throw std::runtime_error("Output allocation failed");
  env->SetFloatArrayRegion(result, 0, output.size(), output.data());
  return result;
}
} // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_org_xrgpu_Native_capabilities(JNIEnv *env, jclass) {
  try {
    return env->NewStringUTF(capabilities().dump().c_str());
  } catch (const std::exception &e) {
    failure(env, e);
    return nullptr;
  }
}
extern "C" JNIEXPORT jlong JNICALL Java_org_xrgpu_Native_create(JNIEnv *env,
                                                                jclass,
                                                                jstring path,
                                                                jint variant) {
  try {
    return reinterpret_cast<jlong>(new Session(string(env, path), variant));
  } catch (const std::exception &e) {
    failure(env, e);
    return 0;
  }
}
extern "C" JNIEXPORT void JNICALL Java_org_xrgpu_Native_destroy(JNIEnv *,
                                                                jclass,
                                                                jlong handle) {
  delete reinterpret_cast<Session *>(handle);
}
extern "C" JNIEXPORT jboolean JNICALL Java_org_xrgpu_Native_supportsLinear(
    JNIEnv *env, jclass, jlong handle, jint rows, jint columns, jint outputs) {
  try {
    const auto &s = session(handle);
    return s.profile && s.profile->linear &&
           s.profile->linear->variant(s.variant) &&
           s.profile->linear->supports(1, rows, columns, outputs, true);
  } catch (const std::exception &e) {
    failure(env, e);
    return false;
  }
}
extern "C" JNIEXPORT jfloatArray JNICALL Java_org_xrgpu_Native_tokens(
    JNIEnv *env, jclass, jlong handle, jlongArray input, jlongArray mask) {
  try {
    const jsize n = env->GetArrayLength(input);
    if ((n != 32 && n != 128 && n != 256) || env->GetArrayLength(mask) != n)
      throw std::runtime_error("Invalid token shapes");
    std::vector<int64_t> ids(n), masks(n), types(n, 0);
    env->GetLongArrayRegion(input, 0, n, reinterpret_cast<jlong *>(ids.data()));
    env->GetLongArrayRegion(mask, 0, n,
                            reinterpret_cast<jlong *>(masks.data()));
    auto a =
        make_tensor_ptr({1, n}, ids.data(), executorch::aten::ScalarType::Long);
    auto b = make_tensor_ptr({1, n}, masks.data(),
                             executorch::aten::ScalarType::Long);
    auto c = make_tensor_ptr({1, n}, types.data(),
                             executorch::aten::ScalarType::Long);
    return array(env, session(handle).run({*a, *b, *c}));
  } catch (const std::exception &e) {
    failure(env, e);
    return nullptr;
  }
}
extern "C" JNIEXPORT jfloatArray JNICALL
Java_org_xrgpu_Native_layer(JNIEnv *env, jclass, jlong handle,
                            jfloatArray input, jint rows, jint columns) {
  try {
    if (rows <= 0 || columns <= 0 ||
        int64_t(rows) * columns != env->GetArrayLength(input))
      throw std::runtime_error("Invalid layer input shape");
    std::vector<float> data(static_cast<size_t>(rows) * columns);
    env->GetFloatArrayRegion(input, 0, data.size(), data.data());
    auto tensor = make_tensor_ptr({1, rows, columns}, data.data(),
                                  executorch::aten::ScalarType::Float);
    return array(env, session(handle).run({*tensor}));
  } catch (const std::exception &e) {
    failure(env, e);
    return nullptr;
  }
}
extern "C" JNIEXPORT jstring JNICALL Java_org_xrgpu_Native_stats(JNIEnv *env,
                                                                 jclass,
                                                                 jlong handle) {
  try {
    auto &s = session(handle);
    Json gpu_ms = nullptr;
    double total = 0;
    for (const auto &event : s.trace->gpu)
      total += event["ms"].get<double>();
    if (!s.trace->gpu.empty())
      gpu_ms = total;
    Json input_ms = nullptr, output_ms = nullptr, submit_wait_ms = nullptr;
    for (const auto &event : s.trace->host) {
      const std::string name = event["name"].get<std::string>();
      Json *value = name == "ETVK_COPY_INPUTS"             ? &input_ms
                    : name == "ETVK_COPY_OUTPUTS"          ? &output_ms
                    : name == "ETVK_COMPUTE_GRAPH_EXECUTE" ? &submit_wait_ms
                                                           : nullptr;
      if (value)
        *value = (value->is_null() ? 0.0 : value->get<double>()) +
                 event["ms"].get<double>();
    }
    Json result = {{"model_ms", s.last_ms},
                   {"load_ms", s.load_ms},
                   {"gpu_ms", gpu_ms},
                   {"input_staging_ms", input_ms},
                   {"output_staging_ms", output_ms},
                   {"submit_and_wait_ms", submit_wait_ms},
                   {"submission_only_ms", nullptr},
                   {"device_memory_bytes", nullptr},
                   {"power_watts", nullptr},
                   {"custom_nodes", s.nodes},
                   {"gpu_events", s.trace->gpu},
                   {"host_events", s.trace->host},
                   {"cpu_delegate_events", s.trace->cpu_delegate}};
    return env->NewStringUTF(result.dump().c_str());
  } catch (const std::exception &e) {
    failure(env, e);
    return nullptr;
  }
}
extern "C" JNIEXPORT jstring JNICALL
Java_org_xrgpu_Native_sceneStats(JNIEnv *env, jclass, jboolean reset) {
  return env->NewStringUTF(xrgpu::scene_snapshot(reset).dump().c_str());
}
extern "C" JNIEXPORT jstring JNICALL
Java_org_xrgpu_Native_sceneStatus(JNIEnv *env, jclass) {
  return env->NewStringUTF(xrgpu::scene_snapshot(false, false).dump().c_str());
}
