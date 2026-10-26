#include <jni.h>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <xrgpu/session.h>
namespace {
std::mutex handles_mutex;
std::unordered_map<jlong, std::shared_ptr<xrgpu::Session>> handles;
jlong next_handle = 1;
void failure(JNIEnv *env, const std::exception &error) {
  const bool invalid =
      dynamic_cast<const std::invalid_argument *>(&error) != nullptr;
  env->ThrowNew(env->FindClass(invalid ? "java/lang/IllegalArgumentException"
                                       : "java/lang/IllegalStateException"),
                error.what());
}
std::string string(JNIEnv *env, jstring input) {
  if (!input)
    throw std::invalid_argument("Null string");
  const char *chars = env->GetStringUTFChars(input, nullptr);
  if (!chars)
    throw std::runtime_error("Cannot read string");
  std::string result(chars);
  env->ReleaseStringUTFChars(input, chars);
  return result;
}
std::shared_ptr<xrgpu::Session> lookup(jlong handle) {
  std::lock_guard<std::mutex> lock(handles_mutex);
  auto found = handles.find(handle);
  if (found == handles.end())
    throw std::runtime_error("Closed session");
  return found->second;
}
jsize count(size_t size) {
  if (size > static_cast<size_t>(std::numeric_limits<jsize>::max()))
    throw std::runtime_error("Tensor exceeds Java array limit");
  return static_cast<jsize>(size);
}
} // namespace
extern "C" JNIEXPORT jlong JNICALL Java_org_xrgpu_Session_load0(
    JNIEnv *env, jclass, jstring path, jboolean cpu, jboolean profiling,
    jstring experiment, jboolean repairs, jboolean repairs_only,
    jstring rules) {
  try {
    xrgpu::SessionOptions options;
    options.force_cpu = cpu;
    options.profiling = profiling;
    options.experiment = string(env, experiment);
    options.experimental_repairs = repairs;
    options.repairs_only = repairs_only;
    options.experimental_rules = string(env, rules);
    std::shared_ptr<xrgpu::Session> session =
        xrgpu::Session::load(string(env, path), options);
    std::lock_guard<std::mutex> lock(handles_mutex);
    if (next_handle == std::numeric_limits<jlong>::max())
      throw std::runtime_error("Session handle limit reached");
    const auto handle = next_handle++;
    handles.emplace(handle, std::move(session));
    return handle;
  } catch (const std::exception &error) {
    failure(env, error);
    return 0;
  }
}
extern "C" JNIEXPORT jobjectArray JNICALL Java_org_xrgpu_Session_forward0(
    JNIEnv *env, jclass, jlong handle, jobjectArray inputs) {
  try {
    if (!inputs)
      throw std::invalid_argument("Null inputs");
    auto type = env->FindClass("org/xrgpu/Tensor");
    auto name_field = env->GetFieldID(type, "name", "Ljava/lang/String;");
    auto shape_field = env->GetFieldID(type, "shape", "[J");
    auto dtype_field = env->GetFieldID(type, "type", "I");
    auto data_field = env->GetFieldID(type, "data", "[B");
    std::vector<xrgpu::Tensor> tensors;
    for (jsize i = 0; i < env->GetArrayLength(inputs); ++i) {
      auto object = env->GetObjectArrayElement(inputs, i);
      if (!object)
        throw std::invalid_argument("Null input tensor");
      xrgpu::Tensor tensor;
      auto name = static_cast<jstring>(env->GetObjectField(object, name_field));
      tensor.name = string(env, name);
      auto shape =
          static_cast<jlongArray>(env->GetObjectField(object, shape_field));
      tensor.shape.resize(env->GetArrayLength(shape));
      env->GetLongArrayRegion(shape, 0, count(tensor.shape.size()),
                              reinterpret_cast<jlong *>(tensor.shape.data()));
      const auto dtype = env->GetIntField(object, dtype_field);
      if (dtype != 0 && dtype != 1)
        throw std::invalid_argument("Unsupported tensor dtype");
      tensor.dtype = dtype == 0 ? xrgpu::DType::Float32 : xrgpu::DType::Int64;
      auto data =
          static_cast<jbyteArray>(env->GetObjectField(object, data_field));
      tensor.data.resize(env->GetArrayLength(data));
      env->GetByteArrayRegion(data, 0, count(tensor.data.size()),
                              reinterpret_cast<jbyte *>(tensor.data.data()));
      tensors.push_back(std::move(tensor));
      env->DeleteLocalRef(name);
      env->DeleteLocalRef(shape);
      env->DeleteLocalRef(data);
      env->DeleteLocalRef(object);
    }
    auto outputs = lookup(handle)->forward(tensors);
    auto constructor =
        env->GetMethodID(type, "<init>", "(Ljava/lang/String;[JI[B)V");
    auto result = env->NewObjectArray(count(outputs.size()), type, nullptr);
    if (!result)
      throw std::runtime_error("Cannot allocate output array");
    for (size_t i = 0; i < outputs.size(); ++i) {
      const auto &output = outputs[i];
      auto name = env->NewStringUTF(output.name.c_str());
      auto shape = env->NewLongArray(count(output.shape.size()));
      auto data = env->NewByteArray(count(output.data.size()));
      if (!name || !shape || !data)
        throw std::runtime_error("Cannot allocate output tensor");
      env->SetLongArrayRegion(
          shape, 0, count(output.shape.size()),
          reinterpret_cast<const jlong *>(output.shape.data()));
      env->SetByteArrayRegion(
          data, 0, count(output.data.size()),
          reinterpret_cast<const jbyte *>(output.data.data()));
      auto object =
          env->NewObject(type, constructor, name, shape,
                         output.dtype == xrgpu::DType::Float32 ? 0 : 1, data);
      env->SetObjectArrayElement(result, count(i), object);
      env->DeleteLocalRef(name);
      env->DeleteLocalRef(shape);
      env->DeleteLocalRef(data);
      env->DeleteLocalRef(object);
    }
    return result;
  } catch (const std::exception &error) {
    failure(env, error);
    return nullptr;
  }
}
extern "C" JNIEXPORT jstring JNICALL
Java_org_xrgpu_Session_info0(JNIEnv *env, jclass, jlong handle) {
  try {
    return env->NewStringUTF(lookup(handle)->info().c_str());
  } catch (const std::exception &error) {
    failure(env, error);
    return nullptr;
  }
}
extern "C" JNIEXPORT void JNICALL Java_org_xrgpu_Session_close0(JNIEnv *env,
                                                                jclass,
                                                                jlong handle) {
  try {
    std::shared_ptr<xrgpu::Session> session;
    {
      std::lock_guard<std::mutex> lock(handles_mutex);
      auto found = handles.find(handle);
      if (found == handles.end())
        return;
      session = found->second;
      handles.erase(found);
    }
    session->close();
  } catch (const std::exception &error) {
    failure(env, error);
  }
}
