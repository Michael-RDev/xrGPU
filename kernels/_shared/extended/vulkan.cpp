// SPDX-License-Identifier: Apache-2.0
#include "framework.h"
#include "../sha256.h"
#include "extended_shader_hashes.h"
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <dlfcn.h>
#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
namespace xrgpu::operators::extended {
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
}

struct Operator::Impl {
 detail::PackedInvocation packed;Request request;Options options;
 std::unique_ptr<Context> context;VkDevice device{};VkQueue queue{};
 struct Buffer{VkBuffer handle{};VkDeviceMemory memory{};void* mapped=nullptr;VkDeviceSize size=0;bool coherent=false;};
 std::vector<Buffer> buffers;
 VkDescriptorSetLayout descriptors{};VkDescriptorPool pool{};VkPipelineLayout layout{};VkCommandPool commands{};VkCommandBuffer command{};VkFence fence{};VkQueryPool queries{};
 struct Step{VkPipeline pipeline{};bool owns_pipeline=true;VkDescriptorSet set{};uint32_t x=0,y=0;std::string shader,digest;};std::vector<Step> steps;
 uint64_t allocated=0;bool closed=false,executed=false,poisoned=false,has_gpu=false;double elapsed=0,gpu=0;
 ~Impl(){release();}
 void release()noexcept{if(!device){context.reset();return;}auto& vk=context->vk;vk.vkDeviceWaitIdle(device);if(queries)vk.vkDestroyQueryPool(device,queries,nullptr);if(fence)vk.vkDestroyFence(device,fence,nullptr);if(commands)vk.vkDestroyCommandPool(device,commands,nullptr);for(auto& s:steps)if(s.pipeline&&s.owns_pipeline)vk.vkDestroyPipeline(device,s.pipeline,nullptr);if(layout)vk.vkDestroyPipelineLayout(device,layout,nullptr);if(pool)vk.vkDestroyDescriptorPool(device,pool,nullptr);if(descriptors)vk.vkDestroyDescriptorSetLayout(device,descriptors,nullptr);for(auto& b:buffers){if(b.mapped)vk.vkUnmapMemory(device,b.memory);if(b.handle)vk.vkDestroyBuffer(device,b.handle,nullptr);if(b.memory)vk.vkFreeMemory(device,b.memory,nullptr);}vk.vkDestroyDevice(device,nullptr);device={};context.reset();closed=true;}
 void open()const{if(closed||!device)throw std::runtime_error("Extended native operator is closed");if(poisoned)throw std::runtime_error("Extended native operator failed and must be closed");}
 void buffer(size_t index,const std::vector<uint32_t>& words){auto& b=buffers[index];auto& c=*context;auto& vk=c.vk;b.size=std::max<size_t>(4,words.size()*4);if(b.size>c.properties.limits.maxStorageBufferRange)throw std::runtime_error("Extended heap exceeds device storage limit");VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};ci.size=b.size;ci.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;ci.sharingMode=VK_SHARING_MODE_EXCLUSIVE;checked(vk.vkCreateBuffer(device,&ci,nullptr,&b.handle),"vkCreateBuffer");VkMemoryRequirements req{};vk.vkGetBufferMemoryRequirements(device,b.handle,&req);if(req.size>options.memory_limit_bytes||allocated>options.memory_limit_bytes-req.size)throw std::runtime_error("Extended allocation limit exceeded");allocated+=req.size;uint32_t chosen=UINT32_MAX;for(uint32_t i=0;i<c.memory_properties.memoryTypeCount;++i){auto f=c.memory_properties.memoryTypes[i].propertyFlags;if((req.memoryTypeBits&(uint32_t(1)<<i))&&(f&VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)){chosen=i;if(f&VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)break;}}if(chosen==UINT32_MAX)throw std::runtime_error("Host-visible Vulkan heap unavailable");b.coherent=bool(c.memory_properties.memoryTypes[chosen].propertyFlags&VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};ai.allocationSize=req.size;ai.memoryTypeIndex=chosen;checked(vk.vkAllocateMemory(device,&ai,nullptr,&b.memory),"vkAllocateMemory");checked(vk.vkBindBufferMemory(device,b.handle,b.memory,0),"vkBindBufferMemory");checked(vk.vkMapMemory(device,b.memory,0,VK_WHOLE_SIZE,0,&b.mapped),"vkMapMemory");std::memset(b.mapped,0,size_t(b.size));if(!words.empty())std::memcpy(b.mapped,words.data(),words.size()*4);if(!b.coherent){VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};range.memory=b.memory;range.size=VK_WHOLE_SIZE;checked(vk.vkFlushMappedMemoryRanges(device,1,&range),"vkFlushMappedMemoryRanges");}}
 void invalidate(size_t i)const{auto& b=buffers[i];if(!b.coherent){VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};range.memory=b.memory;range.size=VK_WHOLE_SIZE;checked(context->vk.vkInvalidateMappedMemoryRanges(device,1,&range),"vkInvalidateMappedMemoryRanges");}}
 std::vector<uint32_t> shader(const std::string& package,std::string name,std::string& digest){if(name.size()>4&&name.substr(name.size()-4)==".spv")name.resize(name.size()-4);if(name.empty()||name.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789_")!=std::string::npos)throw std::invalid_argument("Invalid extended shader name");name+=".spv";const char* expected=nullptr;for(auto h:detail::kExtendedShaderHashes)if(name==h.name){expected=h.sha256;break;}if(!expected)throw std::runtime_error("Extended shader missing from library build inventory");auto path=std::filesystem::absolute(package)/"shaders/extended"/name;for(auto p=path;!p.empty();p=p.parent_path()){if(std::filesystem::is_symlink(p))throw std::invalid_argument("Extended shader links forbidden");if(p==p.root_path())break;}digest=xrgpu::file_sha256(path.string());if(digest!=expected)throw std::runtime_error("Extended shader differs from native library inventory");std::ifstream file(path,std::ios::binary|std::ios::ate);auto n=file.tellg();if(n<20||n>64*1024*1024||uint64_t(n)%4)throw std::runtime_error("Invalid extended SPIR-V length");std::vector<uint32_t> code(size_t(n)/4);file.seekg(0);file.read(reinterpret_cast<char*>(code.data()),n);if(!file||code[0]!=0x07230203)throw std::runtime_error("Invalid extended SPIR-V");return code;}
 void initialize(const std::string& package){if(!options.timeout_ns||!options.memory_limit_bytes)throw std::invalid_argument("Positive extended runtime limits required");packed=detail::pack_plan(extended::plan(request),options);context=std::make_unique<Context>();context->select();auto& c=*context;auto& vk=c.vk;if(c.properties.limits.maxComputeWorkGroupInvocations<64||c.properties.limits.maxComputeWorkGroupSize[0]<64||c.properties.limits.maxPerStageDescriptorStorageBuffers<4)throw std::runtime_error("Extended Vulkan compute limits unavailable");float priority=1;VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};qi.queueFamilyIndex=c.family;qi.queueCount=1;qi.pQueuePriorities=&priority;VkPhysicalDeviceFeatures features{};vk.vkGetPhysicalDeviceFeatures(c.physical,&features);VkPhysicalDeviceFeatures enabled{};enabled.robustBufferAccess=features.robustBufferAccess;VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};di.queueCreateInfoCount=1;di.pQueueCreateInfos=&qi;di.pEnabledFeatures=&enabled;checked(vk.vkCreateDevice(c.physical,&di,nullptr,&device),"vkCreateDevice");vk.vkGetDeviceQueue(device,c.family,0,&queue);buffers.resize(3+packed.passes.size());buffer(0,packed.input_words);buffer(1,packed.output_words);buffer(2,packed.scratch_words);for(size_t i=0;i<packed.passes.size();++i){auto params=packed.base_params;params[3]=packed.passes[i].work_items;params[4]=uint32_t(packed.scratch_words.size());params.insert(params.end(),packed.passes[i].args.begin(),packed.passes[i].args.end());buffer(3+i,params);}
 VkDescriptorSetLayoutBinding bindings[4]{};for(uint32_t i=0;i<4;++i){bindings[i].binding=i;bindings[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;bindings[i].descriptorCount=1;bindings[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;}VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};li.bindingCount=4;li.pBindings=bindings;checked(vk.vkCreateDescriptorSetLayout(device,&li,nullptr,&descriptors),"vkCreateDescriptorSetLayout");VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,uint32_t(4*packed.passes.size())};VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};pi.maxSets=uint32_t(packed.passes.size());pi.poolSizeCount=1;pi.pPoolSizes=&ps;checked(vk.vkCreateDescriptorPool(device,&pi,nullptr,&pool),"vkCreateDescriptorPool");VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};pli.setLayoutCount=1;pli.pSetLayouts=&descriptors;checked(vk.vkCreatePipelineLayout(device,&pli,nullptr,&layout),"vkCreatePipelineLayout");
 steps.resize(packed.passes.size());for(size_t i=0;i<steps.size();++i){auto& s=steps[i];const auto& pass=packed.passes[i];s.shader=pass.shader;uint64_t groups=(uint64_t(pass.work_items)+63)/64;s.x=uint32_t(std::min<uint64_t>(groups,c.properties.limits.maxComputeWorkGroupCount[0]));s.y=s.x?uint32_t((groups+s.x-1)/s.x):0;if(s.y>c.properties.limits.maxComputeWorkGroupCount[1]||uint64_t(s.x)*s.y*64>UINT32_MAX)throw std::runtime_error("Extended dispatch dimensions exceeded");VkDescriptorSetAllocateInfo si{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};si.descriptorPool=pool;si.descriptorSetCount=1;si.pSetLayouts=&descriptors;checked(vk.vkAllocateDescriptorSets(device,&si,&s.set),"vkAllocateDescriptorSets");VkDescriptorBufferInfo info[4]{};VkWriteDescriptorSet writes[4]{};size_t ids[4]={0,1,3+i,2};for(uint32_t b=0;b<4;++b){auto& buf=buffers[ids[b]];info[b]={buf.handle,0,buf.size};writes[b]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};writes[b].dstSet=s.set;writes[b].dstBinding=b;writes[b].descriptorCount=1;writes[b].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;writes[b].pBufferInfo=&info[b];}vk.vkUpdateDescriptorSets(device,4,writes,0,nullptr);auto previous=std::find_if(steps.begin(),steps.begin()+i,[&](const Step& prior){return prior.shader==pass.shader&&prior.pipeline;});if(previous!=steps.begin()+i){s.pipeline=previous->pipeline;s.digest=previous->digest;s.owns_pipeline=false;continue;}auto code=shader(package,pass.shader,s.digest);VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};smi.codeSize=code.size()*4;smi.pCode=code.data();VkShaderModule module{};checked(vk.vkCreateShaderModule(device,&smi,nullptr,&module),"vkCreateShaderModule");VkComputePipelineCreateInfo pci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};pci.layout=layout;pci.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};pci.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;pci.stage.module=module;pci.stage.pName="main";auto status=vk.vkCreateComputePipelines(device,VK_NULL_HANDLE,1,&pci,nullptr,&s.pipeline);vk.vkDestroyShaderModule(device,module,nullptr);checked(status,"vkCreateComputePipelines");}
 VkCommandPoolCreateInfo ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};ci.queueFamilyIndex=c.family;ci.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;checked(vk.vkCreateCommandPool(device,&ci,nullptr,&commands),"vkCreateCommandPool");VkCommandBufferAllocateInfo ca{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};ca.commandPool=commands;ca.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;ca.commandBufferCount=1;checked(vk.vkAllocateCommandBuffers(device,&ca,&command),"vkAllocateCommandBuffers");VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};checked(vk.vkCreateFence(device,&fi,nullptr,&fence),"vkCreateFence");if(options.profiling&&c.queue_properties.timestampValidBits&&c.properties.limits.timestampComputeAndGraphics&&c.properties.limits.timestampPeriod>0){VkQueryPoolCreateInfo q{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};q.queryType=VK_QUERY_TYPE_TIMESTAMP;q.queryCount=uint32_t(2*steps.size());checked(vk.vkCreateQueryPool(device,&q,nullptr,&queries),"vkCreateQueryPool");}
 packed.input_words.clear();packed.input_words.shrink_to_fit();packed.output_words.clear();packed.output_words.shrink_to_fit();packed.scratch_words.clear();packed.scratch_words.shrink_to_fit();for(auto& t:packed.plan.inputs){t.data.clear();t.data.shrink_to_fit();}for(auto& t:request.inputs){t.data.clear();t.data.shrink_to_fit();}
 }
 void execute(){open();executed=false;has_gpu=false;auto& vk=context->vk;std::memset(buffers[2].mapped,0,size_t(buffers[2].size));if(!buffers[2].coherent){VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};range.memory=buffers[2].memory;range.size=VK_WHOLE_SIZE;checked(vk.vkFlushMappedMemoryRanges(device,1,&range),"vkFlushMappedMemoryRanges");}auto start=Clock::now();try{checked(vk.vkResetFences(device,1,&fence),"vkResetFences");checked(vk.vkResetCommandBuffer(command,0),"vkResetCommandBuffer");VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};checked(vk.vkBeginCommandBuffer(command,&begin),"vkBeginCommandBuffer");if(queries)vk.vkCmdResetQueryPool(command,queries,0,uint32_t(steps.size()*2));VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};barrier.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT|VK_ACCESS_SHADER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;for(size_t i=0;i<steps.size();++i){auto& s=steps[i];vk.vkCmdPipelineBarrier(command,VK_PIPELINE_STAGE_HOST_BIT|VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&barrier,0,nullptr,0,nullptr);vk.vkCmdBindPipeline(command,VK_PIPELINE_BIND_POINT_COMPUTE,s.pipeline);vk.vkCmdBindDescriptorSets(command,VK_PIPELINE_BIND_POINT_COMPUTE,layout,0,1,&s.set,0,nullptr);if(queries)vk.vkCmdWriteTimestamp(command,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,queries,uint32_t(i*2));vk.vkCmdDispatch(command,s.x,s.y,1);if(queries)vk.vkCmdWriteTimestamp(command,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,queries,uint32_t(i*2+1));}barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_HOST_READ_BIT;vk.vkCmdPipelineBarrier(command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&barrier,0,nullptr,0,nullptr);checked(vk.vkEndCommandBuffer(command),"vkEndCommandBuffer");VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};submit.commandBufferCount=1;submit.pCommandBuffers=&command;checked(vk.vkQueueSubmit(queue,1,&submit,fence),"vkQueueSubmit");auto spent=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start).count();if(uint64_t(spent)>=options.timeout_ns)throw std::runtime_error("Extended native submission timed out");checked(vk.vkWaitForFences(device,1,&fence,VK_TRUE,options.timeout_ns-uint64_t(spent)),"vkWaitForFences");elapsed=std::chrono::duration<double,std::milli>(Clock::now()-start).count();invalidate(2);if(static_cast<const uint32_t*>(buffers[2].mapped)[packed.base_params[15]])throw std::runtime_error("GPU operator contract validation failed: code "+std::to_string(static_cast<const uint32_t*>(buffers[2].mapped)[packed.base_params[15]]));if(queries){std::vector<uint64_t> times(steps.size()*2);checked(vk.vkGetQueryPoolResults(device,queries,0,uint32_t(times.size()),times.size()*8,times.data(),8,VK_QUERY_RESULT_64_BIT),"vkGetQueryPoolResults");auto bits=context->queue_properties.timestampValidBits;auto mask=bits>=64?UINT64_MAX:((uint64_t(1)<<bits)-1);gpu=0;for(size_t i=0;i<steps.size();++i)gpu+=double((times[2*i+1]-times[2*i])&mask)*context->properties.limits.timestampPeriod/1e6;has_gpu=true;}executed=true;}catch(...){poisoned=true;throw;}}
};
Operator::Operator(std::unique_ptr<Impl> p):impl_(std::move(p)){}
Operator::~Operator()=default;
std::unique_ptr<Operator> Operator::prepare(const std::string& path,const Request& request,Options options){auto p=std::make_unique<Impl>();p->request=request;p->options=options;p->initialize(path);return std::unique_ptr<Operator>(new Operator(std::move(p)));}
void Operator::run(){impl_->execute();}
std::vector<Tensor> Operator::read_outputs()const{auto& p=*impl_;p.open();if(!p.executed)throw std::runtime_error("Run extended operator before readback");p.invalidate(1);p.invalidate(2);return detail::unpack_outputs(p.packed,static_cast<const uint32_t*>(p.buffers[1].mapped),static_cast<const uint32_t*>(p.buffers[2].mapped),p.request.requested_outputs);}
void Operator::close(){impl_->release();impl_->closed=true;}
std::string Operator::info()const{auto& p=*impl_;std::ostringstream o;o.precision(17);o<<"{\"backend\":"<<detail::json_quote(kBackend)<<",\"test_only\":"<<(kTestOnly?"true":"false")<<",\"physical_qualification\":false,\"cpu_compute_fallback\":false,\"extended\":true,\"operation\":"<<detail::json_quote(p.request.operation)<<",\"closed\":"<<(p.closed?"true":"false")<<",\"executed\":"<<(p.executed?"true":"false")<<",\"execution_ms\":";if(p.executed)o<<p.elapsed;else o<<"null";o<<",\"gpu_ms\":";if(p.has_gpu)o<<p.gpu;else o<<"null";o<<",\"dispatch_count\":"<<(p.executed?p.steps.size():0)<<",\"persistent_bytes\":"<<p.allocated<<",\"device\":"<<(p.context?detail::json_quote(p.context->properties.deviceName):"null")<<",\"shaders\":[";for(size_t i=0;i<p.steps.size();++i){if(i)o<<',';o<<"{\"name\":"<<detail::json_quote(p.steps[i].shader)<<",\"sha256\":"<<detail::json_quote(p.steps[i].digest)<<"}";}o<<"]}";return o.str();}
}
