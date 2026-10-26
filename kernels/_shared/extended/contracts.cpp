// SPDX-License-Identifier: Apache-2.0
#include "framework.h"
#include "contract_data.h"
#include <map>
#include <set>
namespace xrgpu::operators::extended::detail {
static const TensorContract* entry(const std::string& operation,const std::string& name,bool output){for(const auto& c:kTensorContracts)if(operation==c.operation&&name==c.name&&output==c.output)return &c;return nullptr;}
static void check_tensor(const TensorContract& c,const Tensor& t,std::map<std::string,DType>& types){
 if(uint32_t(t.dtype)>uint32_t(DType::Bool)||(c.dtype_mask&(1u<<uint32_t(t.dtype)))==0)throw std::invalid_argument("Logical dtype outside frozen contract: "+t.name);
 if(c.rank>=0&&t.shape.size()!=size_t(c.rank))throw std::invalid_argument("Rank outside frozen contract: "+t.name);
 if(c.variable[0]>='A'&&c.variable[0]<='Z'){auto prior=types.find(c.variable);if(prior!=types.end()&&prior->second!=t.dtype)throw std::invalid_argument("Type variable differs between tensors: "+std::string(c.variable));types[c.variable]=t.dtype;}
}
static bool native_variadic(const std::string& op,const std::string& name){return name.size()==1&&name[0]>='a'&&name[0]<='z'&&(op=="ai.onnx.Sum"||op=="ai.onnx.Mean"||op=="ai.onnx.Max"||op=="ai.onnx.Min");}
void validate_contract(const Request& r,const std::vector<Tensor>* outputs){
 std::map<std::string,DType> types;std::set<std::string> provided;
 for(const auto& t:r.inputs){auto c=entry(r.operation,t.name,false);if(!c&&native_variadic(r.operation,t.name))c=entry(r.operation,"a",false);if(!c)throw std::invalid_argument("Unknown input outside frozen contract: "+t.name);check_tensor(*c,t,types);provided.insert(t.name);}
 for(const auto& c:kTensorContracts)if(r.operation==c.operation&&!c.output&&!c.optional&&!provided.count(c.name))throw std::invalid_argument("Missing required frozen-contract input: "+std::string(c.name));
 if(!outputs)return;provided.clear();for(const auto& t:*outputs){auto c=entry(r.operation,t.name,true);if(!c)throw std::invalid_argument("Unknown output outside frozen contract: "+t.name);check_tensor(*c,t,types);provided.insert(t.name);}
 for(const auto& c:kTensorContracts)if(r.operation==c.operation&&c.output&&!c.optional&&!provided.count(c.name))throw std::invalid_argument("Missing required frozen-contract output: "+std::string(c.name));
}
}
