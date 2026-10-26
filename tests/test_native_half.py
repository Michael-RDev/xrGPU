"""Bit-exact host checks of the original GLSL binary16 storage conversion."""
import ctypes
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def convert(tmp_path_factory):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("Host C++ compiler required")
    root = tmp_path_factory.mktemp("native-half")
    source = root / "half.cpp"
    source.write_text(r"""
#include <cstdint>
#include <cstring>
#include <cstddef>
using uint=uint32_t;
float uintBitsToFloat(uint x){float y;std::memcpy(&y,&x,4);return y;}
uint floatBitsToUint(float x){uint y;std::memcpy(&y,&x,4);return y;}
""" + '#include "' + str(ROOT / 'kernels/_shared/shaders/half.glsl') + '"\n' + r"""
extern "C" void decode(const uint* in,uint* out,size_t size){for(size_t i=0;i<size;++i)out[i]=floatBitsToUint(xrgpuHalfToFloat(in[i]));}
extern "C" void encode(const uint* in,uint* out,size_t size){for(size_t i=0;i<size;++i)out[i]=xrgpuFloatToHalf(uintBitsToFloat(in[i]));}
""")
    library = root / "half.dylib"
    subprocess.run([compiler, "-std=c++17", "-O2", "-shared", "-fPIC", str(source), "-o", str(library)], check=True, capture_output=True)
    loaded = ctypes.CDLL(str(library))
    for name in ("encode", "decode"):
        function = getattr(loaded, name)
        function.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t]
        function.restype = None

    def invoke(name, bits):
        bits = np.ascontiguousarray(bits, dtype=np.uint32)
        output = np.empty_like(bits)
        getattr(loaded, name)(bits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)), output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)), len(bits))
        return output
    return invoke


def test_all_65536_half_patterns_decode(convert):
    bits = np.arange(65536, dtype=np.uint16)
    values = bits.view(np.float16)
    actual = convert("decode", bits).view(np.float32)
    with np.errstate(all="ignore"):
        expected = values.astype(np.float32)
    finite_or_inf = ~np.isnan(expected)
    np.testing.assert_array_equal(actual[finite_or_inf].view(np.uint32), expected[finite_or_inf].view(np.uint32))
    assert np.all(np.isnan(actual[~finite_or_inf]))
    np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))


def test_half_encode_every_rounding_boundary_and_neighbor(convert):
    half = np.arange(0x7c00, dtype=np.uint16).view(np.float16).astype(np.float32)
    midpoints = (half[:-1] + half[1:]) / np.float32(2)
    values = np.concatenate([midpoints, np.nextafter(midpoints, np.float32(np.inf)),
                             np.nextafter(midpoints, np.float32(-np.inf)),
                             np.array([65504, 65519.99609375, 65520, 65536, 70000, 1e10, np.inf], dtype=np.float32)])
    values = np.r_[values, -values].astype(np.float32)
    with np.errstate(all="ignore"):
        expected = values.astype(np.float16).view(np.uint16)
    actual = convert("encode", values.view(np.uint32))
    np.testing.assert_array_equal(actual, expected)


def test_half_encode_random_float32_patterns_and_nan_classification(convert):
    rng = np.random.default_rng(64129)
    bits = rng.integers(0, 2**32, 100003, dtype=np.uint32)
    values = bits.view(np.float32)
    with np.errstate(all="ignore"):
        expected = values.astype(np.float16).view(np.uint16)
    actual = convert("encode", bits).astype(np.uint16)
    finite_or_inf = ~np.isnan(values)
    np.testing.assert_array_equal(actual[finite_or_inf], expected[finite_or_inf])
    assert np.all((actual[~finite_or_inf] & 0x7c00) == 0x7c00)
    assert np.all((actual[~finite_or_inf] & 0x03ff) != 0)
    assert np.all((actual[~finite_or_inf] & 0x0200) != 0)
    np.testing.assert_array_equal(actual >> 15, bits >> 31)
