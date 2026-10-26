"""Numerical checks of the original GLSL scalar algorithms, executed on host CPU.

These exercise the exact shared shader function as C++ with host math functions;
they cannot establish physical GPU conformance or device float-control behavior.
The independent software-Vulkan runner covers compiled SPIR-V separately.
"""

import ctypes
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from kernels import catalog, provenance

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "kernels/_shared/shaders/unary_ops.glsl"
DTYPES = {0: ("float32", 0, 32), 1: ("float16", 1, 16), 2: ("int32", 2, 32),
          3: ("uint32", 3, 32), 4: ("int8", 5, 8), 5: ("uint8", 6, 8), 6: ("int16", 4, 16)}


def registrations():
    return [(row["id"], row["slug"], row["opcode"], row["dtype_mask"],
             row["input_names"][0], row["output_name"])
            for row in catalog.legacy_records()
            if row["kind"] == "Unary" and row["opcode"] >= 32]


@pytest.fixture(scope="module")
def scalar(tmp_path_factory):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("A host C++ compiler is required")
    directory = tmp_path_factory.mktemp("native-unary")
    prelude = r'''
#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>
#define precise
using uint = uint32_t;
struct vec2 { float x,y; vec2():x(0),y(0) {} vec2(float a,float b):x(a),y(b) {} };
void umulExtended(uint a,uint b,uint& high,uint& low) {
  uint64_t product=uint64_t(a)*b;high=uint(product>>32);low=uint(product);
}
using std::isnan; using std::isinf; using std::fma;
using std::acos; using std::acosh; using std::asin; using std::asinh;
using std::atan; using std::atanh; using std::ceil; using std::cos; using std::cosh;
using std::exp; using std::floor; using std::log; using std::sin; using std::sinh;
using std::sqrt; using std::tan; using std::tanh;
float max(float a, float b) { return std::fmax(a,b); }
float roundEven(float x) { return std::nearbyint(x); }
float uintBitsToFloat(uint v) { float x; std::memcpy(&x,&v,4); return x; }
uint floatBitsToUint(float x) { uint v; std::memcpy(&v,&x,4); return v; }
'''
    chunks = [prelude]
    dispatch = []
    for op, slug, opcode, mask, _, _ in registrations() + [("ai.onnx.Relu", "relu", 1, 55, "x", "y")]:
        for dtype, (_, bit, bits) in DTYPES.items():
            if dtype == 1 or not mask & (1 << bit):
                continue
            namespace = f"kernel_{opcode}_{dtype}"
            chunks.append(f'namespace {namespace} {{\n#define DTYPE {dtype}\n#define OPCODE {opcode}\n#define BITS {bits}u\n#define MASK {(1<<bits)-1}u\n')
            chunks.append('float decodeFloat(uint a) { return uintBitsToFloat(a); }\nuint encodeFloat(float x) { return floatBitsToUint(x); }\n')
            chunks.append(f'#include "{SOURCE}"\n#undef DTYPE\n#undef OPCODE\n#undef BITS\n#undef MASK\n}}\n')
            dispatch.append(f'if(opcode=={opcode} && dtype=={dtype}) return {namespace}::evaluateUnary(value);')
    chunks.append('extern "C" uint evaluate(int opcode,int dtype,uint value) {\n' + '\n'.join(dispatch) + '\nreturn 0; }\n')
    source = directory / "scalar.cpp"
    source.write_text(''.join(chunks))
    library = directory / "scalar.dylib"
    subprocess.run([compiler, "-std=c++17", "-shared", "-fPIC", "-O2", str(source), "-o", str(library)], check=True, capture_output=True)
    module = ctypes.CDLL(str(library))
    module.evaluate.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint32]
    module.evaluate.restype = ctypes.c_uint32
    return module.evaluate


def run(scalar, opcode, values, dtype=0):
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    return np.array([scalar(opcode, dtype, int(v)) for v in bits], dtype=np.uint32).view(np.float32)


REFERENCES = {
    "abs": np.abs, "acos": np.arccos, "acosh": np.arccosh, "asin": np.arcsin,
    "asinh": np.arcsinh, "atan": np.arctan, "atanh": np.arctanh, "ceil": np.ceil,
    "cos": np.cos, "cosh": np.cosh, "erf": np.vectorize(math.erf), "exp": np.exp,
    "floor": np.floor, "log": np.log, "neg": np.negative, "reciprocal": np.reciprocal,
    "round": np.rint, "sign": np.sign, "sin": np.sin, "sinh": np.sinh, "sqrt": np.sqrt,
    "tan": np.tan, "tanh": np.tanh, "sigmoid": lambda x: np.exp(-np.logaddexp(0, -x)),
    "softplus": lambda x: np.logaddexp(0, x), "softsign": lambda x: x/(1+np.abs(x)),
    "mish": lambda x: x*np.tanh(np.logaddexp(0, x)),
    "hardswish": lambda x: x*np.clip((x+3)/6, 0, 1),
}


@pytest.mark.parametrize("op,slug,opcode,mask,input_name,output", registrations())
def test_unary_scalar_math_against_independent_references(scalar, op, slug, opcode, mask, input_name, output):
    rng = np.random.default_rng(41731)
    values = np.r_[rng.uniform(-5,5,501), -0.0, 0.0, -0.5, 0.5, -1.5, 1.5,
                   -100.0, 100.0, -np.inf, np.inf, np.nan].astype(np.float32)
    with np.errstate(all="ignore"):
        expected = np.asarray(REFERENCES[slug](values), dtype=np.float32)
        actual = run(scalar, opcode, values)
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-7, equal_nan=True, err_msg=op)


def test_bit_exact_abs_neg_sign_relu_preserve_subnormal_input_information(scalar):
    bits = np.array([0x00000001,0x80000001,0x007fffff,0x807fffff,0x7fc00001,0xffc00001],dtype=np.uint32)
    assert [scalar(32,0,int(v)) for v in bits] == [int(v&0x7fffffff) for v in bits]
    assert [scalar(46,0,int(v)) for v in bits] == [int(v^0x80000000) for v in bits]
    assert [scalar(49,0,int(v)) for v in bits] == [0x3f800000,0xbf800000,0x3f800000,0xbf800000,0x7fc00001,0xffc00001]
    assert [scalar(1,0,int(v)) for v in bits] == [1,0,0x007fffff,0,0x7fc00001,0xffc00001]


def test_rounding_and_reciprocal_signed_zero_boundaries(scalar):
    values = np.array([-0.0,0.0,-0.5,0.5,-1.5,1.5,-2.5,2.5],dtype=np.float32)
    rounded = run(scalar,48,values)
    np.testing.assert_array_equal(rounded,np.rint(values))
    np.testing.assert_array_equal(np.signbit(rounded),np.signbit(np.rint(values)))
    boundaries = np.array([-0.0,0.0,-np.inf,np.inf],dtype=np.float32)
    expected = np.array([-np.inf,np.inf,-0.0,0.0],dtype=np.float32)
    np.testing.assert_array_equal(run(scalar,47,boundaries).view(np.uint32),expected.view(np.uint32))


@pytest.mark.parametrize("dtype,bits",[(2,32),(4,8),(6,16)])
def test_signed_integer_sign_and_minimum_wrapping(scalar,dtype,bits):
    mask = (1<<bits)-1
    minimum = 1<<(bits-1)
    assert scalar(49,dtype,minimum)==mask
    assert scalar(49,dtype,mask)==mask
    assert scalar(49,dtype,0)==0
    assert scalar(49,dtype,1)==1
    if dtype != 6:
        assert scalar(32,dtype,minimum)==minimum
        assert scalar(46,dtype,minimum)==minimum
        assert scalar(46,dtype,mask)==1


def test_native_registry_uses_only_existing_contract_types_and_names():
    for op,slug,opcode,mask,input_name,output in registrations():
        contract = provenance.document(op, "contract")
        assert list(contract["inputs"]) == [input_name]
        assert list(contract["outputs"]) == [output]
        allowed = set(contract["typeConstraints"][contract["inputs"][input_name]["dtype"]])
        declared = {name for name,bit,bits in DTYPES.values() if mask&(1<<bit)}
        assert declared <= allowed
        assert not contract.get("attributes")


def test_trig_range_reduction_handles_all_float32_exponents(scalar):
    rng = np.random.default_rng(81173)
    bits = rng.integers(0,2**32,size=10001,dtype=np.uint32)
    values = bits.view(np.float32)
    values = values[np.isfinite(values)]
    values = np.r_[values,np.array([1e6,1e7,1e13,1e20,-1e15,-123456.78,314159.265,2.5e9,
                                   65504,9999,10000,10001],dtype=np.float32)]
    for opcode,reference in ((50,np.sin),(40,np.cos)):
        actual = run(scalar,opcode,values)
        expected = reference(values.astype(np.float64)).astype(np.float32)
        np.testing.assert_allclose(actual,expected,rtol=2e-7,atol=2e-7,equal_nan=True)


def test_inverse_trig_half_angle_edges(scalar):
    values = np.array([-1,-0.9999999403953552,-0.5,-0.0,0,0.5,0.9999999403953552,1],dtype=np.float32)
    for opcode,reference in ((33,np.arccos),(35,np.arcsin)):
        np.testing.assert_allclose(run(scalar,opcode,values),reference(values),rtol=2e-7,atol=2e-7)


def test_exponential_and_activation_negative_tails(scalar):
    values=np.array([-100,-90,-88,-80,-30,-20,-19,-18,-17,-16,-9,0,10,80,87,88,88.5,89],dtype=np.float32)
    with np.errstate(all="ignore"):
        expected=np.exp(values.astype(np.float64)).astype(np.float32)
        np.testing.assert_allclose(run(scalar,43,values),expected,rtol=1e-7,atol=1e-45)
    tails=np.array([-90,-88,-80,-30,-20,-19,-18,-17,-16,-9],dtype=np.float32)
    expected=np.logaddexp(0,tails.astype(np.float64)).astype(np.float32)
    np.testing.assert_allclose(run(scalar,56,tails),expected,rtol=2e-7,atol=1e-44)
    expected=(tails.astype(np.float64)*np.tanh(np.logaddexp(0,tails.astype(np.float64)))).astype(np.float32)
    np.testing.assert_allclose(run(scalar,58,tails),expected,rtol=2e-7,atol=1e-44)


def test_logarithm_normalizes_subnormals_by_bits(scalar):
    values=np.array([1,2,0x7fffff,0x800000,0x3f800000,0x7f7fffff],dtype=np.uint32).view(np.float32)
    np.testing.assert_array_equal(run(scalar,45,values),np.log(values.astype(np.float64)).astype(np.float32))


def test_activation_extremes_have_ieee_classification(scalar):
    values=np.array([-np.inf,-np.finfo(np.float32).max,-100,-0.0,0.0,100,np.finfo(np.float32).max,np.inf],dtype=np.float32)
    np.testing.assert_array_equal(run(scalar,54,values),np.tanh(values))
    actual=run(scalar,59,values)
    assert np.isnan(actual[0])
    assert actual[-1]==np.inf
    assert actual[1]==0 and np.signbit(actual[1])


def test_trig_double_float_path_rounds_large_argument_cases(scalar):
    rng=np.random.default_rng(93179)
    values=rng.integers(0,2**32,size=12001,dtype=np.uint32).view(np.float32)
    values=values[np.isfinite(values)]
    values=np.r_[values,np.array([1e6,1e7,1e13,1e20,-1e15,-123456.78,314159.265,2.5e9],dtype=np.float32)]
    for opcode,reference in ((50,np.sin),(40,np.cos),(53,np.tan)):
        expected=reference(values.astype(np.float64)).astype(np.float32)
        np.testing.assert_array_equal(run(scalar,opcode,values),expected)


def test_mish_and_softplus_preserve_residual_until_output(scalar):
    rng=np.random.default_rng(94439)
    values=np.r_[rng.uniform(-17,16,10001),-2.0,0.0,2.0].astype(np.float32)
    working=values.astype(np.float64)
    softplus=np.logaddexp(0,working)
    expected_mish=(working*np.tanh(softplus)).astype(np.float32)
    np.testing.assert_array_equal(run(scalar,58,values),expected_mish)
    np.testing.assert_array_equal(run(scalar,56,values),softplus.astype(np.float32))
