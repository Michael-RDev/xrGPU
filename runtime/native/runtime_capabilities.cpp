#include "runtime_capabilities.h"
#ifdef __ANDROID__
#include <sys/system_properties.h>
#endif
#ifdef XRGPU_HAS_VULKAN
#include <dlfcn.h>
#include <vector>
#include <vulkan/vulkan.h>
#endif
namespace xrgpu {
nlohmann::json runtime_capabilities() {
  nlohmann::json result = {{"vulkan", false}, {"model", "host"}};
#ifdef __ANDROID__
  char model[PROP_VALUE_MAX]{};
  __system_property_get("ro.product.model", model);
  result["model"] = model;
#endif
#ifdef XRGPU_HAS_VULKAN
  void *library = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
  if (!library) {
    result["reason"] = "Vulkan loader unavailable";
    return result;
  }
  auto proc = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
      dlsym(library, "vkGetInstanceProcAddr"));
  if (!proc) {
    dlclose(library);
    result["reason"] = "Vulkan entry point unavailable";
    return result;
  }
  auto create = reinterpret_cast<PFN_vkCreateInstance>(
      proc(VK_NULL_HANDLE, "vkCreateInstance"));
  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "xrGPU";
  app.apiVersion = VK_API_VERSION_1_1;
  VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  info.pApplicationInfo = &app;
  VkInstance instance = VK_NULL_HANDLE;
  if (!create || create(&info, nullptr, &instance) != VK_SUCCESS) {
    dlclose(library);
    result["reason"] = "Vulkan 1.1 unavailable";
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
  if (enumerate && properties && destroy &&
      enumerate(instance, &count, nullptr) == VK_SUCCESS && count) {
    std::vector<VkPhysicalDevice> devices(count);
    if (enumerate(instance, &count, devices.data()) == VK_SUCCESS && count) {
      VkPhysicalDeviceProperties p{};
      properties(devices.front(), &p);
      bool timestamps = false;
      if (queues) {
        uint32_t count = 0;
        queues(devices.front(), &count, nullptr);
        std::vector<VkQueueFamilyProperties> properties(count);
        queues(devices.front(), &count, properties.data());
        for (const auto &family : properties)
          if (family.queueFlags & VK_QUEUE_COMPUTE_BIT) {
            timestamps = family.timestampValidBits > 0;
            break;
          }
      }
      result.update(
          {{"vulkan", true},
           {"gpu", p.deviceName},
           {"vendor_id", p.vendorID},
           {"device_id", p.deviceID},
           {"driver", p.driverVersion},
           {"api_version", p.apiVersion},
           {"gpu_timestamps", timestamps && p.limits.timestampPeriod > 0}});
    }
  }
  if (destroy)
    destroy(instance, nullptr);
  dlclose(library);
#else
  result["reason"] = "Built without Vulkan";
#endif
  return result;
}
} // namespace xrgpu
