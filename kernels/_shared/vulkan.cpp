// SPDX-License-Identifier: Apache-2.0
#include "internal.h"
#include "sha256.h"
#include "shader_hashes.h"
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <dlfcn.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace xrgpu::operators {
namespace {
using Clock=std::chrono::steady_clock;
#ifdef XRGPU_OPERATOR_TEST_SOFTWARE_VULKAN
constexpr const char* kBackend="software_vulkan";
constexpr bool kTestOnly=true;
#else
constexpr const char* kBackend="xrgpu_vulkan";
constexpr bool kTestOnly=false;
#endif
void checked(VkResult status,const char* operation) {
  if(status!=VK_SUCCESS)throw std::runtime_error(std::string(operation)+" failed: Vulkan status "+std::to_string(status));
}
#define XRGPU_VK_FUNCTIONS(X) \
 X(vkCreateInstance) X(vkDestroyInstance) X(vkEnumeratePhysicalDevices) \
 X(vkGetPhysicalDeviceProperties) X(vkGetPhysicalDeviceFeatures) X(vkGetPhysicalDeviceQueueFamilyProperties) \
 X(vkGetPhysicalDeviceMemoryProperties) X(vkCreateDevice) X(vkDestroyDevice) X(vkGetDeviceQueue) \
 X(vkCreateBuffer) X(vkDestroyBuffer) X(vkGetBufferMemoryRequirements) X(vkAllocateMemory) X(vkFreeMemory) \
 X(vkBindBufferMemory) X(vkMapMemory) X(vkUnmapMemory) X(vkFlushMappedMemoryRanges) X(vkInvalidateMappedMemoryRanges) \
 X(vkCreateDescriptorSetLayout) X(vkDestroyDescriptorSetLayout) X(vkCreateDescriptorPool) X(vkDestroyDescriptorPool) \
 X(vkAllocateDescriptorSets) X(vkUpdateDescriptorSets) X(vkCreatePipelineLayout) X(vkDestroyPipelineLayout) \
 X(vkCreateShaderModule) X(vkDestroyShaderModule) X(vkCreateComputePipelines) X(vkDestroyPipeline) \
 X(vkCreateCommandPool) X(vkDestroyCommandPool) X(vkAllocateCommandBuffers) X(vkResetCommandBuffer) \
 X(vkBeginCommandBuffer) X(vkEndCommandBuffer) X(vkCmdPipelineBarrier) X(vkCmdBindPipeline) \
 X(vkCmdBindDescriptorSets) X(vkCmdDispatch) X(vkCreateFence) X(vkDestroyFence) X(vkResetFences) \
 X(vkQueueSubmit) X(vkWaitForFences) X(vkDeviceWaitIdle) X(vkCreateQueryPool) X(vkDestroyQueryPool) \
 X(vkCmdResetQueryPool) X(vkCmdWriteTimestamp) X(vkGetQueryPoolResults)
struct Dispatch {
 void* library=nullptr;
#define DECLARE(name) PFN_##name name=nullptr;
 XRGPU_VK_FUNCTIONS(DECLARE)
#undef DECLARE
 Dispatch() {
#ifdef XRGPU_OPERATOR_TEST_SOFTWARE_VULKAN
  const char* path=std::getenv("XRGPU_TEST_VULKAN_LIBRARY");
  library=dlopen(path?path:"libvulkan.dylib",RTLD_NOW|RTLD_LOCAL);
#elif defined(__ANDROID__)
  library=dlopen("libvulkan.so",RTLD_NOW|RTLD_LOCAL);
#elif defined(__linux__)
  library=dlopen("libvulkan.so.1",RTLD_NOW|RTLD_LOCAL);
#else
  throw std::runtime_error("Native Vulkan loader unavailable on this platform");
#endif
  if(!library)throw std::runtime_error("Native Vulkan loader unavailable");
  try {
#define LOAD(name) name=reinterpret_cast<PFN_##name>(dlsym(library,#name));if(!name)throw std::runtime_error("Missing Vulkan entry point: " #name);
   XRGPU_VK_FUNCTIONS(LOAD)
#undef LOAD
  } catch(...) {dlclose(library);library=nullptr;throw;}
 }
 ~Dispatch(){if(library)dlclose(library);}
};
struct Context {
 Dispatch vk;VkInstance instance{};VkPhysicalDevice physical{};VkPhysicalDeviceProperties properties{};
 VkPhysicalDeviceMemoryProperties memory_properties{};VkQueueFamilyProperties queue_properties{};uint32_t family=0;
 ~Context(){if(instance)vk.vkDestroyInstance(instance,nullptr);}
 void select() {
  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};app.pApplicationName="xrGPU Native Operators";app.apiVersion=VK_API_VERSION_1_1;
  VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};ci.pApplicationInfo=&app;
  checked(vk.vkCreateInstance(&ci,nullptr,&instance),"vkCreateInstance");
  uint32_t count=0;checked(vk.vkEnumeratePhysicalDevices(instance,&count,nullptr),"vkEnumeratePhysicalDevices");
  std::vector<VkPhysicalDevice> devices(count);
  if(count)checked(vk.vkEnumeratePhysicalDevices(instance,&count,devices.data()),"vkEnumeratePhysicalDevices");
  for(auto candidate:devices) {
   VkPhysicalDeviceProperties props{};vk.vkGetPhysicalDeviceProperties(candidate,&props);
   if(props.apiVersion<VK_API_VERSION_1_1)continue;
#ifndef XRGPU_OPERATOR_TEST_SOFTWARE_VULKAN
   if(props.deviceType==VK_PHYSICAL_DEVICE_TYPE_CPU)continue;
#endif
   uint32_t n=0;vk.vkGetPhysicalDeviceQueueFamilyProperties(candidate,&n,nullptr);
   std::vector<VkQueueFamilyProperties> families(n);vk.vkGetPhysicalDeviceQueueFamilyProperties(candidate,&n,families.data());
   for(uint32_t i=0;i<n;++i) if(families[i].queueCount&&(families[i].queueFlags&VK_QUEUE_COMPUTE_BIT)) {
    physical=candidate;properties=props;family=i;queue_properties=families[i];
    vk.vkGetPhysicalDeviceMemoryProperties(physical,&memory_properties);return;
   }
  }
  throw std::runtime_error("No Vulkan 1.1 GPU compute device; CPU Vulkan devices are not accepted");
 }
};
std::string tensor_json(const Tensor& t) {
 std::ostringstream o;o<<"{\"name\":"<<detail::json_quote(t.name)<<",\"dtype\":"<<detail::json_quote(dtype_name(t.dtype))
 <<",\"storage_dtype\":"<<detail::json_quote(dtype_name(detail::storage(t)))<<",\"layout\":\"contiguous\",\"shape\":[";
 for(size_t i=0;i<t.shape.size();++i){if(i)o<<',';o<<t.shape[i];}o<<"]}";return o.str();
}
}
struct Operator::Impl {
 std::unique_ptr<Context> context;detail::Plan plan;Options options;VkDevice device{};VkQueue queue{};
 struct Buffer {VkBuffer handle{};VkDeviceMemory memory{};void* mapped=nullptr;VkDeviceSize size=0;bool coherent=false;};
 std::array<Buffer,4> buffers{};uint64_t allocated_bytes=0;
 VkDescriptorSetLayout descriptors{};VkDescriptorPool descriptor_pool{};VkDescriptorSet descriptor_set{};
 VkPipelineLayout pipeline_layout{};VkPipeline pipeline{};VkCommandPool command_pool{};VkCommandBuffer command{};
 VkFence fence{};VkQueryPool queries{};uint32_t groups_x=0,groups_y=0;bool executed=false,closed=false,poisoned=false;
 double execution_ms=0,gpu_ms=0;bool has_gpu_time=false;std::string shader_digest;
 ~Impl(){release();}
 void release() noexcept {
  if(!device){context.reset();return;}
  auto& vk=context->vk;
  vk.vkDeviceWaitIdle(device);
  if(queries)vk.vkDestroyQueryPool(device,queries,nullptr);
  if(fence)vk.vkDestroyFence(device,fence,nullptr);
  if(command_pool)vk.vkDestroyCommandPool(device,command_pool,nullptr);
  if(pipeline)vk.vkDestroyPipeline(device,pipeline,nullptr);
  if(pipeline_layout)vk.vkDestroyPipelineLayout(device,pipeline_layout,nullptr);
  if(descriptor_pool)vk.vkDestroyDescriptorPool(device,descriptor_pool,nullptr);
  if(descriptors)vk.vkDestroyDescriptorSetLayout(device,descriptors,nullptr);
  for(auto& b:buffers){if(b.mapped)vk.vkUnmapMemory(device,b.memory);if(b.handle)vk.vkDestroyBuffer(device,b.handle,nullptr);if(b.memory)vk.vkFreeMemory(device,b.memory,nullptr);b={};}
  vk.vkDestroyDevice(device,nullptr);device={};context.reset();closed=true;
 }
 void check_open() const {if(closed||!device)throw std::runtime_error("Native operator is closed");if(poisoned)throw std::runtime_error("Native operator failed and must be closed");}
 void flush(Buffer& b) {
  if(!b.coherent){VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};range.memory=b.memory;range.size=VK_WHOLE_SIZE;checked(context->vk.vkFlushMappedMemoryRanges(device,1,&range),"vkFlushMappedMemoryRanges");}
 }
 void buffer(size_t index,uint64_t bytes,const uint8_t* input=nullptr,uint64_t input_bytes=0) {
  auto& b=buffers[index];auto& c=*context;auto& vk=c.vk;
  b.size=std::max<uint64_t>(4,(bytes+3)&~uint64_t(3));
  if(b.size>c.properties.limits.maxStorageBufferRange)throw std::runtime_error("Unsupported storage buffer range");
  VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};ci.size=b.size;ci.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;ci.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
  checked(vk.vkCreateBuffer(device,&ci,nullptr,&b.handle),"vkCreateBuffer");
  VkMemoryRequirements req{};vk.vkGetBufferMemoryRequirements(device,b.handle,&req);
  if(req.size>options.memory_limit_bytes||allocated_bytes>options.memory_limit_bytes-req.size)throw std::runtime_error("Operator memory limit exceeded");
  allocated_bytes+=req.size;uint32_t chosen=UINT32_MAX;
  for(uint32_t i=0;i<c.memory_properties.memoryTypeCount;++i) {
   auto flags=c.memory_properties.memoryTypes[i].propertyFlags;
   if((req.memoryTypeBits&(uint32_t(1)<<i))&&(flags&VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
    chosen=i;if(flags&VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)break;
   }
  }
  if(chosen==UINT32_MAX)throw std::runtime_error("Unsupported GPU memory: host-visible storage required");
  b.coherent=bool(c.memory_properties.memoryTypes[chosen].propertyFlags&VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};ai.allocationSize=req.size;ai.memoryTypeIndex=chosen;
  checked(vk.vkAllocateMemory(device,&ai,nullptr,&b.memory),"vkAllocateMemory");
  checked(vk.vkBindBufferMemory(device,b.handle,b.memory,0),"vkBindBufferMemory");
  checked(vk.vkMapMemory(device,b.memory,0,VK_WHOLE_SIZE,0,&b.mapped),"vkMapMemory");
  std::memset(b.mapped,0,size_t(b.size));if(input_bytes)std::memcpy(b.mapped,input,size_t(input_bytes));flush(b);
 }
 void initialize(const std::string& package) {
  if(!options.timeout_ns||!options.memory_limit_bytes)throw std::invalid_argument("Positive native runtime limits required");
  // Shader filenames are selected exclusively by the validated typed plan.
  auto root=std::filesystem::absolute(package);
  auto shader=root/"shaders"/plan.shader;
  for(auto p=shader;!p.empty();p=p.parent_path()) {
   if(std::filesystem::is_symlink(p))throw std::invalid_argument("Shader package links are forbidden");
   if(p==p.root_path())break;
  }
  const char* expected=nullptr;
  for(const auto& record:detail::kShaderHashes)if(plan.shader==record.name){expected=record.sha256;break;}
  if(!expected)throw std::runtime_error("Shader was not inventoried in this library build");
  shader_digest=xrgpu::file_sha256(shader.string());
  if(shader_digest!=expected)throw std::runtime_error("Compiled shader differs from native library inventory");
  std::ifstream file(shader,std::ios::binary|std::ios::ate);auto length=file.tellg();
  if(length<20||length>16*1024*1024||uint64_t(length)%4)throw std::runtime_error("Invalid SPIR-V binary length");
  std::vector<uint32_t> code(size_t(length)/4);file.seekg(0);file.read(reinterpret_cast<char*>(code.data()),length);
  if(!file||code[0]!=0x07230203)throw std::runtime_error("Invalid SPIR-V binary");
  for(const auto& t:plan.inputs)if(t.data.size()!=detail::elements(t)*dtype_bytes(detail::storage(t)))throw std::invalid_argument("Input tensor byte count mismatch: "+t.name);
  context=std::make_unique<Context>();context->select();auto& c=*context;auto& vk=c.vk;
  if(c.properties.limits.maxComputeWorkGroupInvocations<64||c.properties.limits.maxComputeWorkGroupSize[0]<64||c.properties.limits.maxPerStageDescriptorStorageBuffers<4)
   throw std::runtime_error("Unsupported Vulkan compute limits");
  uint64_t groups=(uint64_t(plan.output_words)+63)/64;
  groups_x=uint32_t(std::min<uint64_t>(groups,c.properties.limits.maxComputeWorkGroupCount[0]));
  groups_y=groups_x?uint32_t((groups+groups_x-1)/groups_x):0;
  if(uint64_t(groups_x)*groups_y*64>UINT32_MAX||groups_y>c.properties.limits.maxComputeWorkGroupCount[1])throw std::runtime_error("Unsupported Vulkan dispatch dimensions");
  float priority=1;VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};qi.queueFamilyIndex=c.family;qi.queueCount=1;qi.pQueuePriorities=&priority;
  VkPhysicalDeviceFeatures supported{};vk.vkGetPhysicalDeviceFeatures(c.physical,&supported);VkPhysicalDeviceFeatures enabled{};enabled.robustBufferAccess=supported.robustBufferAccess;
  VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};di.queueCreateInfoCount=1;di.pQueueCreateInfos=&qi;di.pEnabledFeatures=&enabled;
  checked(vk.vkCreateDevice(c.physical,&di,nullptr,&device),"vkCreateDevice");vk.vkGetDeviceQueue(device,c.family,0,&queue);
  buffer(0,plan.inputs[0].data.size(),plan.inputs[0].data.data(),plan.inputs[0].data.size());
  if(plan.inputs.size()==2)buffer(1,plan.inputs[1].data.size(),plan.inputs[1].data.data(),plan.inputs[1].data.size());else buffer(1,4);
  buffer(2,uint64_t(plan.output_words)*4);
  buffer(3,sizeof(plan.parameters),reinterpret_cast<const uint8_t*>(plan.parameters.data()),sizeof(plan.parameters));
  VkDescriptorSetLayoutBinding bindings[4]{};
  for(uint32_t i=0;i<4;++i){bindings[i].binding=i;bindings[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;bindings[i].descriptorCount=1;bindings[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;}
  VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};li.bindingCount=4;li.pBindings=bindings;
  checked(vk.vkCreateDescriptorSetLayout(device,&li,nullptr,&descriptors),"vkCreateDescriptorSetLayout");
  VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,4};VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};pi.maxSets=1;pi.poolSizeCount=1;pi.pPoolSizes=&ps;
  checked(vk.vkCreateDescriptorPool(device,&pi,nullptr,&descriptor_pool),"vkCreateDescriptorPool");
  VkDescriptorSetAllocateInfo si{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};si.descriptorPool=descriptor_pool;si.descriptorSetCount=1;si.pSetLayouts=&descriptors;
  checked(vk.vkAllocateDescriptorSets(device,&si,&descriptor_set),"vkAllocateDescriptorSets");
  VkDescriptorBufferInfo bi[4]{};VkWriteDescriptorSet writes[4]{};
  for(uint32_t i=0;i<4;++i){bi[i]={buffers[i].handle,0,buffers[i].size};writes[i]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};writes[i].dstSet=descriptor_set;writes[i].dstBinding=i;writes[i].descriptorCount=1;writes[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;writes[i].pBufferInfo=&bi[i];}
  vk.vkUpdateDescriptorSets(device,4,writes,0,nullptr);
  VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};pli.setLayoutCount=1;pli.pSetLayouts=&descriptors;
  checked(vk.vkCreatePipelineLayout(device,&pli,nullptr,&pipeline_layout),"vkCreatePipelineLayout");
  VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};smi.codeSize=code.size()*4;smi.pCode=code.data();VkShaderModule module{};
  checked(vk.vkCreateShaderModule(device,&smi,nullptr,&module),"vkCreateShaderModule");
  VkComputePipelineCreateInfo pci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};pci.layout=pipeline_layout;pci.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};pci.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;pci.stage.module=module;pci.stage.pName="main";
  auto status=vk.vkCreateComputePipelines(device,VK_NULL_HANDLE,1,&pci,nullptr,&pipeline);vk.vkDestroyShaderModule(device,module,nullptr);checked(status,"vkCreateComputePipelines");
  VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};cpi.queueFamilyIndex=c.family;cpi.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  checked(vk.vkCreateCommandPool(device,&cpi,nullptr,&command_pool),"vkCreateCommandPool");
  VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};cai.commandPool=command_pool;cai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;cai.commandBufferCount=1;
  checked(vk.vkAllocateCommandBuffers(device,&cai,&command),"vkAllocateCommandBuffers");
  VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};checked(vk.vkCreateFence(device,&fi,nullptr,&fence),"vkCreateFence");
  if(options.profiling&&c.queue_properties.timestampValidBits&&c.properties.limits.timestampComputeAndGraphics&&c.properties.limits.timestampPeriod>0) {
   VkQueryPoolCreateInfo q{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};q.queryType=VK_QUERY_TYPE_TIMESTAMP;q.queryCount=2;
   checked(vk.vkCreateQueryPool(device,&q,nullptr,&queries),"vkCreateQueryPool");
  }
  // Retain specs only after their bytes have been staged in resident GPU buffers.
  for(auto& t:plan.inputs){t.data.clear();t.data.shrink_to_fit();}
 }
 void execute() {
  check_open();executed=false;has_gpu_time=false;auto& vk=context->vk;const auto start=Clock::now();
  if(!plan.output_words){execution_ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();executed=true;return;}
  try {
   checked(vk.vkResetFences(device,1,&fence),"vkResetFences");checked(vk.vkResetCommandBuffer(command,0),"vkResetCommandBuffer");
   VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};checked(vk.vkBeginCommandBuffer(command,&begin),"vkBeginCommandBuffer");
   if(queries)vk.vkCmdResetQueryPool(command,queries,0,2);
   VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};barrier.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT|VK_ACCESS_SHADER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;
   vk.vkCmdPipelineBarrier(command,VK_PIPELINE_STAGE_HOST_BIT|VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&barrier,0,nullptr,0,nullptr);
   vk.vkCmdBindPipeline(command,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline);vk.vkCmdBindDescriptorSets(command,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline_layout,0,1,&descriptor_set,0,nullptr);
   if(queries)vk.vkCmdWriteTimestamp(command,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,queries,0);
   vk.vkCmdDispatch(command,groups_x,groups_y,1);
   if(queries)vk.vkCmdWriteTimestamp(command,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,queries,1);
   barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_HOST_READ_BIT;
   vk.vkCmdPipelineBarrier(command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&barrier,0,nullptr,0,nullptr);
   checked(vk.vkEndCommandBuffer(command),"vkEndCommandBuffer");VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};submit.commandBufferCount=1;submit.pCommandBuffers=&command;
   checked(vk.vkQueueSubmit(queue,1,&submit,fence),"vkQueueSubmit");
   auto spent=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start).count();
   if(uint64_t(spent)>=options.timeout_ns)throw std::runtime_error("Native Vulkan submission timed out");
   checked(vk.vkWaitForFences(device,1,&fence,VK_TRUE,options.timeout_ns-uint64_t(spent)),"vkWaitForFences");
   execution_ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
   if(queries) {
    uint64_t times[2]{};checked(vk.vkGetQueryPoolResults(device,queries,0,2,sizeof(times),times,sizeof(uint64_t),VK_QUERY_RESULT_64_BIT),"vkGetQueryPoolResults");
    auto bits=context->queue_properties.timestampValidBits;auto mask=bits>=64?UINT64_MAX:((uint64_t(1)<<bits)-1);
    gpu_ms=double((times[1]-times[0])&mask)*context->properties.limits.timestampPeriod/1e6;has_gpu_time=true;
   }
   executed=true;
  } catch(...) {poisoned=true;throw;}
 }
};
std::string probe() {
 try {Context c;c.select();return "{\"available\":true,\"backend\":"+detail::json_quote(kBackend)+",\"test_only\":"+(kTestOnly?"true":"false")+",\"device\":"+detail::json_quote(c.properties.deviceName)+",\"api_version\":"+std::to_string(c.properties.apiVersion)+",\"physical_qualification\":false}";}
 catch(const std::exception& e){return "{\"available\":false,\"backend\":"+detail::json_quote(kBackend)+",\"test_only\":"+(kTestOnly?"true":"false")+",\"reason\":"+detail::json_quote(e.what())+",\"physical_qualification\":false}";}
}
Operator::Operator(std::unique_ptr<Impl> impl):impl_(std::move(impl)){}
Operator::~Operator()=default;
std::unique_ptr<Operator> Operator::prepare(const std::string& package,const std::string& id,const std::vector<Tensor>& inputs,Options options,const std::vector<Tensor>& requested) {
 auto impl=std::make_unique<Impl>();impl->options=options;impl->plan=detail::make_plan(id,inputs,requested);impl->initialize(package);
 return std::unique_ptr<Operator>(new Operator(std::move(impl)));
}
void Operator::run(){impl_->execute();}
std::vector<Tensor> Operator::read_outputs() const {
 auto& s=*impl_;s.check_open();if(!s.executed)throw std::runtime_error("Run the native operator before reading outputs");
 if(!s.buffers[2].coherent){VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};range.memory=s.buffers[2].memory;range.size=VK_WHOLE_SIZE;checked(s.context->vk.vkInvalidateMappedMemoryRanges(s.device,1,&range),"vkInvalidateMappedMemoryRanges");}
 Tensor output=s.plan.output;output.data.resize(detail::elements(output)*dtype_bytes(detail::storage(output)));
 if(!output.data.empty())std::memcpy(output.data.data(),s.buffers[2].mapped,output.data.size());return {std::move(output)};
}
std::string Operator::info() const {
 auto& s=*impl_;std::ostringstream o;o.precision(17);
 o<<"{\"backend\":"<<detail::json_quote(kBackend)<<",\"test_only\":"<<(kTestOnly?"true":"false")<<",\"operation\":"<<detail::json_quote(s.plan.operation)<<",\"closed\":"<<(s.closed?"true":"false")
 <<",\"executed\":"<<(s.executed?"true":"false")<<",\"physical_qualification\":false,\"cpu_compute_fallback\":false,\"native_fp16\":false,\"execution_ms\":";
 if(s.executed)o<<s.execution_ms;else o<<"null";o<<",\"gpu_ms\":";if(s.has_gpu_time)o<<s.gpu_ms;else o<<"null";
 o<<",\"dispatch_count\":"<<(s.executed&&s.plan.output_words?1:0)<<",\"persistent_bytes\":"<<s.allocated_bytes
 <<",\"shader\":"<<detail::json_quote(s.plan.shader)<<",\"shader_sha256\":"<<detail::json_quote(s.shader_digest)
 <<",\"device\":"<<(s.context?detail::json_quote(s.context->properties.deviceName):"null")<<",\"inputs\":[";
 for(size_t i=0;i<s.plan.inputs.size();++i){if(i)o<<',';o<<tensor_json(s.plan.inputs[i]);}
 o<<"],\"outputs\":["<<tensor_json(s.plan.output)<<"]}";return o.str();
}
void Operator::close(){impl_->release();impl_->closed=true;}
} // namespace xrgpu::operators
