"""Full-width native GPU cast algorithms against independent NumPy references."""
import ctypes
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cast(tmp_path_factory):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("Host compiler required")
    directory = tmp_path_factory.mktemp("native-int64")
    source = directory / "casts.cpp"
    source.write_text(r"""
#include <cstdint>
#include <cstddef>
#include <cstring>
using uint=uint32_t;
struct uvec2{uint x=0,y=0;uvec2()=default;uvec2(uint a,uint b=0):x(a),y(b){}};
uint failure=0;
void fail(uint code){failure=code;}
int findMSB(uint x){return x?31-__builtin_clz(x):-1;}
float uintBitsToFloat(uint x){float y;std::memcpy(&y,&x,4);return y;}
uint floatBitsToUint(float x){uint y;std::memcpy(&y,&x,4);return y;}
""" + '#include "' + str(ROOT / 'kernels/_shared/extended/shaders/int64.glsl') + '"\n' + r"""
extern "C" void to_float(const uint64_t* in,uint* out,size_t size){for(size_t i=0;i<size;++i)out[i]=floatBitsToUint(extendedInt64ToFloat(uvec2(uint(in[i]),uint(in[i]>>32))));}
extern "C" void from_float(const uint* in,uint64_t* out,uint* errors,size_t size){for(size_t i=0;i<size;++i){failure=0;uvec2 v=extendedFloatToInt64(uintBitsToFloat(in[i]));out[i]=uint64_t(v.x)|(uint64_t(v.y)<<32);errors[i]=failure;}}
""")
    library = directory / "casts.dylib"
    subprocess.run([compiler, "-std=c++17", "-O2", "-shared", "-fPIC", str(source), "-o", str(library)], check=True, capture_output=True)
    module = ctypes.CDLL(str(library))
    module.to_float.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t]
    module.from_float.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t]
    return module


def to_float(cast, values):
    values = np.ascontiguousarray(values, np.int64)
    output = np.empty(values.shape, np.uint32)
    cast.to_float(values.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)), values.size)
    return output.view(np.float32)


def from_float(cast, values):
    values = np.ascontiguousarray(values, np.float32)
    output = np.empty(values.shape, np.uint64)
    errors = np.empty(values.shape, np.uint32)
    cast.from_float(values.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)), output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), errors.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)), values.size)
    return output.view(np.int64), errors


def test_pair_int64_casts_signed_boundaries_and_large_values(cast):
    values = np.array([0,1,-1,2**31,-2**31,2**32+1,-2**32-1,2**40+3,-2**40-3,2**63-1,-2**63],np.int64)
    np.testing.assert_array_equal(to_float(cast,values).view(np.uint32), values.astype(np.float32).view(np.uint32))


def test_pair_int64_to_float_random_and_every_exponent_rounding(cast):
    rng = np.random.default_rng(7109)
    values = rng.integers(0,2**64,100003,dtype=np.uint64).view(np.int64)
    boundaries = []
    for leading in range(24,63):
        shift = leading-23
        for significand in [0,1,7,0x100001,0x400000]:
            midpoint=(1<<leading)+(significand<<shift)+(1<<(shift-1))
            for adjustment in [-1,0,1]:
                boundaries.extend([midpoint+adjustment,-midpoint-adjustment])
    values = np.r_[values,np.array(boundaries,np.int64)]
    np.testing.assert_array_equal(to_float(cast,values).view(np.uint32),values.astype(np.float32).view(np.uint32))


def test_float_to_pair_int64_truncates_full_width_and_validates_domain(cast):
    rng=np.random.default_rng(8227)
    values=rng.integers(0,2**32,100003,dtype=np.uint32).view(np.float32)
    values=values[np.isfinite(values)]
    values=values[(values.astype(np.float64)>=-(2.0**63))&(values.astype(np.float64)<2.0**63)]
    values=np.r_[values,np.array([3e9,-3e9,2.0**40,-2.0**40,-2.0**63,-2.75,2.75],np.float32)]
    actual,errors=from_float(cast,values)
    assert not np.any(errors)
    np.testing.assert_array_equal(actual,values.astype(np.int64))
    _,errors=from_float(cast,np.array([np.nan,np.inf,-np.inf,2.0**63,-2.0**64],np.float32))
    assert np.all(errors==3)
