// SPDX-License-Identifier: Apache-2.0
#include "internal.h"
#include "registry.h"
#include <algorithm>
#include <limits>
#include <sstream>
#include <set>
namespace xrgpu::operators {
std::string dtype_name(DType t) {
  switch(t) {
    case DType::Float32: return "float32"; case DType::Float16: return "float16";
    case DType::Int32: return "int32"; case DType::UInt32: return "uint32";
    case DType::Int16: return "int16"; case DType::Int8: return "int8";
    case DType::UInt8: return "uint8"; case DType::Int64: return "int64";
    case DType::Bool: return "bool"; case DType::Unspecified: return "unspecified";
  }
  throw std::invalid_argument("Unknown dtype");
}
DType parse_dtype(const std::string& name) {
  for(auto t:{DType::Float32,DType::Float16,DType::Int32,DType::UInt32,DType::Int16,
              DType::Int8,DType::UInt8,DType::Int64,DType::Bool,DType::Unspecified})
    if(dtype_name(t)==name) return t;
  throw std::invalid_argument("Unsupported dtype: "+name);
}
uint32_t dtype_bytes(DType t) {
  switch(t) {
    case DType::Float32: case DType::Int32: case DType::UInt32: return 4;
    case DType::Float16: case DType::Int16: return 2;
    case DType::Int8: case DType::UInt8: case DType::Bool: return 1;
    case DType::Int64: return 8;
    default: throw std::invalid_argument("A concrete dtype is required");
  }
}
namespace detail {
DType storage(const Tensor& t) { return t.storage_dtype==DType::Unspecified?t.dtype:t.storage_dtype; }
uint64_t elements(const Tensor& t) {
  if(t.shape.size()>kMaxRank) throw std::invalid_argument("Unsupported tensor rank: alpha supports at most 16");
  bool empty=false;
  for(auto d:t.shape) {
    if(d<0 || uint64_t(d)>UINT32_MAX) throw std::invalid_argument("Invalid tensor dimension");
    empty=empty||d==0;
  }
  if(empty) return 0;
  uint64_t n=1;
  for(auto d:t.shape) {
    if(n>UINT32_MAX/uint64_t(d)) throw std::invalid_argument("Tensor exceeds uint32 addressable elements");
    n*=uint64_t(d);
  }
  return n;
}
static void validate(const Tensor& t) {
  elements(t);
  if(t.layout!="contiguous") throw std::invalid_argument("Unsupported layout: contiguous row-major storage required");
  if(storage(t)!=t.dtype) throw std::invalid_argument("Unsupported physical storage dtype for this operator");
  dtype_bytes(t.dtype);
}
static std::vector<int64_t> broadcast(const std::vector<int64_t>& a,const std::vector<int64_t>& b) {
  std::vector<int64_t> out(std::max(a.size(),b.size()),1);
  for(size_t i=0;i<out.size();++i) {
    int64_t x=i<a.size()?a[a.size()-1-i]:1,y=i<b.size()?b[b.size()-1-i]:1;
    if(x!=y && x!=1 && y!=1) throw std::invalid_argument("Input shapes do not broadcast");
    out[out.size()-1-i]=x==1?y:x;
  }
  return out;
}
static void strides(const std::vector<int64_t>& input,const std::vector<int64_t>& out,
                    std::array<uint32_t,kParameterWords>& p,uint32_t offset,
                    uint64_t initial=1) {
  uint64_t stride=initial;
  for(size_t reverse=0;reverse<out.size();++reverse) {
    const auto at=out.size()-1-reverse;
    if(reverse<input.size()) {
      auto d=input[input.size()-1-reverse];
      if(d!=1) {
        if(stride>UINT32_MAX) throw std::invalid_argument("Tensor stride exceeds uint32 range");
        p[offset+at]=uint32_t(stride);
      }
      if(d!=0 && stride>UINT32_MAX/uint64_t(d)) {
        // No element will be addressed if the complete input is empty.
        stride=uint64_t(UINT32_MAX)+1;
      } else stride*=uint64_t(d);
    }
  }
}
std::string json_quote(const std::string& s) {
  std::ostringstream o;o<<'"';
  constexpr char hex[]="0123456789abcdef";
  for(unsigned char c:s) {
    if(c=='"'||c=='\\') o<<'\\'<<char(c);
    else if(c<32) o<<"\\u00"<<hex[c>>4]<<hex[c&15];
    else o<<char(c);
  }
  o<<'"';return o.str();
}
Plan make_plan(const std::string& id,const std::vector<Tensor>& inputs,const std::vector<Tensor>& requested) {
  Plan result;result.operation=id;
  const auto& record=operation_record(id);
  const bool relu=record.kind==Kind::Unary,add=record.kind==Kind::Binary;
  const std::vector<std::string> names=relu?std::vector<std::string>{record.input_a}:std::vector<std::string>{record.input_a,record.input_b};
  if(inputs.size()!=names.size()) throw std::invalid_argument("Incorrect input count");
  for(const auto& name:names) {
    auto it=std::find_if(inputs.begin(),inputs.end(),[&](const Tensor& t){return t.name==name;});
    if(it==inputs.end()) throw std::invalid_argument("Missing required input: "+name);
    validate(*it);result.inputs.push_back(*it);
  }
  auto type=result.inputs[0].dtype;
  for(const auto& t:result.inputs) if(t.dtype!=type) throw std::invalid_argument("Input logical dtypes must match");
  if((record.dtype_mask&(uint32_t(1)<<static_cast<unsigned>(type)))==0)
    throw std::invalid_argument("Unsupported dtype for "+id+": "+dtype_name(type));
  result.output.name=record.output;result.output.dtype=type;result.output.storage_dtype=type;
  auto& p=result.parameters;
  if(relu) result.output.shape=result.inputs[0].shape;
  else if(add) {
    result.output.shape=broadcast(result.inputs[0].shape,result.inputs[1].shape);
    if(id=="ai.onnx.PRelu"&&result.output.shape!=result.inputs[0].shape)
      throw std::invalid_argument("PRelu slope must broadcast into x without expanding x");
    p[1]=uint32_t(result.output.shape.size());
    for(size_t i=0;i<result.output.shape.size();++i)p[8+i]=uint32_t(result.output.shape[i]);
    if(elements(result.output)) {
      strides(result.inputs[0].shape,result.output.shape,p,8+kMaxRank);
      strides(result.inputs[1].shape,result.output.shape,p,8+2*kMaxRank);
    }
  } else {
    auto a=result.inputs[0].shape,b=result.inputs[1].shape;
    if(a.empty()||b.empty()) throw std::invalid_argument("MatMul inputs require rank at least one");
    bool av=a.size()==1,bv=b.size()==1;
    if(av)a.insert(a.begin(),1);
    if(bv)b.push_back(1);
    auto m=a[a.size()-2],k=a.back(),n=b.back();
    if(k!=b[b.size()-2])throw std::invalid_argument("MatMul inner dimensions differ");
    std::vector<int64_t> ab(a.begin(),a.end()-2),bb(b.begin(),b.end()-2);
    auto batch=broadcast(ab,bb);
    result.output.shape=batch;if(!av)result.output.shape.push_back(m);if(!bv)result.output.shape.push_back(n);
    p[1]=uint32_t(batch.size());p[2]=uint32_t(m);p[3]=uint32_t(n);p[4]=uint32_t(k);
    for(size_t i=0;i<batch.size();++i)p[8+i]=uint32_t(batch[i]);
    if(elements(result.output)&&k) {
      strides(ab,batch,p,8+kMaxRank,uint64_t(m)*k);
      strides(bb,batch,p,8+2*kMaxRank,uint64_t(k)*n);
    }
  }
  p[0]=uint32_t(elements(result.output));
  uint64_t bytes=uint64_t(p[0])*dtype_bytes(type);result.output_words=uint32_t((bytes+3)/4);
  if(requested.size()>1)throw std::invalid_argument("This operator produces exactly one output");
  if(!requested.empty()) {
    const auto& t=requested[0];validate(t);
    if(t.name!=result.output.name||t.dtype!=type||t.shape!=result.output.shape)
      throw std::invalid_argument("Requested output metadata differs from inferred contract");
    if(!t.data.empty())throw std::invalid_argument("Requested output must be a descriptor without data");
  }
  std::string tag;
  switch(type) {
    case DType::Float32: tag="f32";break;case DType::Float16:tag="f16";break;
    case DType::Int32:tag="i32";break;case DType::UInt32:tag="u32";break;
    case DType::Int16:tag="i16";break;case DType::Int8:tag="i8";break;
    case DType::UInt8:tag="u8";break;default:throw std::invalid_argument("Unsupported dtype");
  }
  result.shader=std::string(record.slug)+"_"+tag+".spv";
  return result;
}
} // namespace detail
std::vector<Tensor> plan(const std::string& id,const std::vector<Tensor>& inputs,const std::vector<Tensor>& out) {
  return {detail::make_plan(id,inputs,out).output};
}
} // namespace xrgpu::operators
