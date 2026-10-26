"""Independent cache-format/window tests for the exact native GLSL helper.

The shared helper executes as C++ for exhaustive storage/layout checks; the
complex-family Vulkan fixture suite separately executes its compiled SPIR-V.
"""
import ctypes
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("Host compiler required")
    directory = tmp_path_factory.mktemp("native-cache")
    source = directory / "cache.cpp"
    prelude = r"""
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include "HALF_HOST"
using uint=uint32_t;
struct uvec2 {uint x,y;uvec2(uint a,uint b=0):x(a),y(b){}};
constexpr uint D_F32=0,D_F16=1,D_I8=5,D_U8=6;
struct Slot{uint type=0;std::vector<uint> shape,data;};
Slot in[15],out[3];uint args[128],failure;
std::vector<uint> outputHeap;
using std::isnan;using std::isinf;
uint os(uint canonical){return canonical;}
uint arg(uint index){return args[index];}
void fail(uint code){failure=std::max(failure,code);}
bool has(uint slot){return !in[slot].shape.empty();}
uint rank(uint slot){return in[slot].shape.size();}
uint dim(uint slot,uint axis){return in[slot].shape[axis];}
uint size(uint slot){return in[slot].data.size();}
uint dtype(uint slot){return in[slot].type;}
uint outRank(uint slot){return out[slot].shape.size();}
uint outDim(uint slot,uint axis){return out[slot].shape[axis];}
uint outDtype(uint slot){return out[slot].type;}
uint readU(uint slot,uint index){return in[slot].data.at(index);}
int readI(uint slot,uint index){uint x=readU(slot,index);return in[slot].type==D_I8?int(x<<24)>>24:int(x);}
float readF(uint slot,uint index){uint x=readU(slot,index);if(in[slot].type==D_F16)return xrgpu::operators::extended::detail::half_host::xrgpuHalfToFloat(x);float value;std::memcpy(&value,&x,4);return value;}
uvec2 readRaw(uint slot,uint index){return uvec2(readU(slot,index));}
uint outputAddress(uint slot,uint index){return (slot-1)*out[1].data.size()+index;}
void writeU(uint slot,uint index,uint value){outputHeap.at(outputAddress(slot,index))=value;}
void writeRaw(uint slot,uint index,uvec2 value){writeU(slot,index,value.x);}
void writeF(uint slot,uint index,float value){uint bits;if(out[slot].type==D_F16)bits=xrgpu::operators::extended::detail::half_host::xrgpuFloatToHalf(value);else std::memcpy(&bits,&value,4);writeU(slot,index,bits);}
float clamp(float x,float lo,float hi){return std::fmin(std::fmax(x,lo),hi);}
float roundEven(float x){return std::nearbyint(x);}
uint atomicCompSwap(uint& location,uint old,uint replacement){uint value=location;if(value==old)location=replacement;return value;}
#include "CACHE_HELPER"
extern "C" int run(uint role,uint type,uint bits,uint mode,uint window,uint batches,uint heads,uint capacity,uint outputCapacity,uint added,uint width,const uint* past,const uint* input,const float* scales,const float* current,uint* result,float* decoded){
 for(auto& s:in)s=Slot{};for(auto& s:out)s=Slot{};std::fill(args,args+128,0);failure=0;
 uint physical=bits==4&&(type==D_I8||type==D_U8)?(width+1)/2:width;
 args[2]=heads;args[7]=added;args[8]=args[9]=width;args[21]=bits;args[22]=args[23]=mode;args[24]=window;
 uint source=complexCacheInput(role),scaleSlot=complexCacheScaleInput(role);
 in[source].type=type;in[source].shape={batches,heads,capacity,physical};in[source].data.assign(input,input+batches*heads*capacity*physical);
 in[5].type=2;in[5].shape={batches};for(uint b=0;b<batches;++b)in[5].data.push_back(past[b]+added-1);
 uint scaleCount=mode==2?heads*width:1;in[scaleSlot].type=D_F32;in[scaleSlot].shape={scaleCount};for(uint i=0;i<scaleCount;++i){uint value;std::memcpy(&value,scales+i,4);in[scaleSlot].data.push_back(value);}
 for(uint output=1;output<=2;++output){out[output].type=type;out[output].shape={batches,heads,outputCapacity,physical};out[output].data.resize(batches*heads*outputCapacity*physical);}
 outputHeap.assign(out[1].data.size()*2,0);
 complexCacheCopy(role);
 for(uint b=0;b<batches;++b)for(uint h=0;h<heads;++h)for(uint row=0;row<added;++row)for(uint d=0;d<width;++d)complexCacheWrite(role,b,h,past[b]+row,d,current[((b*heads+h)*added+row)*width+d]);
 std::copy(outputHeap.begin()+(role-1)*out[1].data.size(),outputHeap.begin()+role*out[1].data.size(),result);
 for(uint b=0;b<batches;++b)for(uint h=0;h<heads;++h)for(uint row=0;row<capacity;++row)for(uint d=0;d<width;++d){uint first=window&&past[b]>capacity?past[b]-capacity:0;decoded[((b*heads+h)*capacity+row)*width+d]=first+row<past[b]?complexCacheRead(role,b,h,first+row,d):0.0f;}
 return failure;
}
"""
    prelude = prelude.replace("HALF_HOST", str(ROOT / "kernels/_shared/extended/half_host.h"))
    prelude = prelude.replace("CACHE_HELPER", str(ROOT / "kernels/_shared/extended/shaders/complex_quant_cache.glsl"))
    source.write_text(prelude)
    library = directory / "cache.dylib"
    subprocess.run([compiler, "-std=c++17", "-shared", "-fPIC", "-O2", str(source), "-o", str(library)], check=True, capture_output=True)
    module = ctypes.CDLL(str(library))
    module.run.argtypes = [ctypes.c_uint32] * 11 + [ctypes.POINTER(ctypes.c_uint32)] * 2 + [ctypes.POINTER(ctypes.c_float)] * 2 + [ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_float)]
    module.run.restype = ctypes.c_int

    def invoke(role, dtype, bits, mode, window, old, past, scales, current, output_capacity=None):
        old = np.ascontiguousarray(old, np.uint32)
        past = np.ascontiguousarray(past, np.uint32)
        scales = np.ascontiguousarray(scales, np.float32)
        current = np.ascontiguousarray(current, np.float32)
        batches, heads, capacity, physical = old.shape
        added, width = current.shape[-2:]
        output_capacity = output_capacity or capacity
        result = np.empty((batches, heads, output_capacity, physical), np.uint32)
        decoded = np.empty((batches, heads, capacity, width), np.float32)
        def u(a):
            return a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))

        def f(a):
            return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        code = module.run(role, dtype, bits, mode, window, batches, heads, capacity, output_capacity, added, width, u(past), u(old), f(scales), f(current), u(result), f(decoded))
        return code, result, decoded
    return invoke


def quantize(value, scale, bits):
    with np.errstate(all="ignore"):
        return np.clip(np.rint(np.float32(value) / np.float32(scale)), -(1 << (bits-1)), (1 << (bits-1))-1).astype(np.int32)


@pytest.mark.parametrize("role,values,expected", [
    (1, [-8,-7,-1,0,1,6,7,-2], [16,135,233,111]),
    (2, [7,6,1,0,-1,-7,-8,2], [239,137,23,160]),
])
def test_int4_frozen_biased_nibble_goldens(cache, role, values, expected):
    code, result, _ = cache(role, 6, 4, 1, 0, np.zeros((1,1,1,4), np.uint32), [0], [1], np.array(values).reshape(1,1,1,8))
    assert code == 0
    np.testing.assert_array_equal(result.ravel(), expected)


@pytest.mark.parametrize("bits,mode", [(4,1),(4,2),(8,1),(8,2)])
def test_quantized_copy_decode_and_channel_scaling(cache, bits, mode):
    batches, heads, capacity, width = 2,2,4,8
    rng = np.random.default_rng(811)
    codes = rng.integers(-(1 << (bits-1)), 1 << (bits-1), (batches,heads,capacity,width), dtype=np.int32)
    old = ((codes[...,::2]+8) | ((codes[...,1::2]+8)<<4)).astype(np.uint32) if bits == 4 else (codes & 255).astype(np.uint32)
    scales = np.array([0.5], np.float32) if mode == 1 else np.arange(1,heads*width+1,dtype=np.float32)/8
    current = rng.uniform(-7,7,(batches,heads,2,width)).astype(np.float32)
    past = np.array([2,2],np.uint32)
    code, result, decoded = cache(1, 6 if bits==4 else 5, bits, mode, 0, old, past, scales, current)
    assert code == 0
    expected = old.copy()
    for b in range(batches):
        for h in range(heads):
            channel_scale = scales[0] if mode == 1 else scales[h*width:(h+1)*width]
            np.testing.assert_allclose(decoded[b,h,:2], codes[b,h,:2]*channel_scale, rtol=0, atol=0)
            q = quantize(current[b,h],channel_scale,bits)
            expected[b,h,2:] = (q[:,::2]+8)|((q[:,1::2]+8)<<4) if bits==4 else q&255
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("kind", [0,1,5,6])
def test_sliding_front_eviction_preserves_raw_old_storage(cache, kind):
    width, capacity, added, past = 8,4,2,9
    bits = 4 if kind == 6 else 8
    old_values = np.arange(capacity*width,dtype=np.float32).reshape(1,1,capacity,width)/4
    if kind == 0:
        old = old_values.view(np.uint32)
    elif kind == 1:
        old = old_values.astype(np.float16).view(np.uint16).astype(np.uint32)
    elif bits == 8:
        old = np.arange(capacity*width,dtype=np.uint32).reshape(1,1,capacity,width)&255
    else:
        old = np.arange(capacity*width//2,dtype=np.uint32).reshape(1,1,capacity,width//2)
    current = np.array([np.arange(width),-np.arange(width)],np.float32).reshape(1,1,added,width)
    code, result, _ = cache(2,kind,bits,0 if kind<2 else 1,1,old,[past],[1],current)
    assert code == 0
    np.testing.assert_array_equal(result[0,0,:2],old[0,0,2:])
    if kind == 0:
        expected = current.view(np.uint32)
    elif kind == 1:
        expected = current.astype(np.float16).view(np.uint16).astype(np.uint32)
    else:
        q = quantize(current,1,bits)
        expected = q&255 if bits==8 else (q[...,::2]+8)|((q[...,1::2]+8)<<4)
    np.testing.assert_array_equal(result[0,0,2:],expected[0,0])


def test_int8_round_even_saturation_and_untouched_padding(cache):
    old = np.full((1,1,4,8), 77, np.uint32)
    values = np.array([-1000,-128.5,-127.5,-0.5,0.5,1.5,126.5,1000],np.float32).reshape(1,1,1,8)
    code, actual, _ = cache(1,5,8,1,0,old,[1],[1],values)
    assert code == 0
    np.testing.assert_array_equal(actual[0,0,0],old[0,0,0])
    np.testing.assert_array_equal(actual[0,0,2:],old[0,0,2:])
    np.testing.assert_array_equal(actual[0,0,1],quantize(values.ravel(),1,8)&255)
