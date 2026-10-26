# SPDX-License-Identifier: Apache-2.0
"""Oracle and typed-fixture checks; GPU execution is recorded separately."""

import numpy as np
from kernels._shared.extended.complex_reference import (
    advanced_cases,
    feature_cases,
    linear_attention_reference,
    moe_reference,
    published_cases,
    representatives,
)


def test_representatives_cover_36_real_operations():
    cases = representatives()
    assert len(cases) == len({case["operation"] for case in cases}) == 36
    assert all(
        any(value.size and np.any(value) for value in case["expected"].values()) for case in cases
    )


def test_frozen_fixture_preserves_logical_int64_storage_int32():
    case = next(
        case for case in published_cases() if case["operation"] == "com.microsoft.MRotaryEmbedding"
    )
    assert case["input_descriptors"]["positionIds"]["dtype"] == "int64"
    assert case["inputs"]["positionIds"].dtype == np.int32


def test_dense_and_quantized_routing_use_distinct_documented_softmax_domains():
    dense = dict(
        inputT=np.array([[1]], np.float32),
        routerT=np.array([[0, 0]], np.float32),
        fc1T=np.array([[[1]], [[2]]], np.float32),
        fc2T=np.array([[[1]], [[3]]], np.float32),
    )
    assert moe_reference(dense, {})[0, 0] == 3
    assert moe_reference(dense, dict(normalize_routing_weights=1))[0, 0] == 6
    quantized = dict(
        inputT=np.array([[1]], np.float32),
        routerT=np.array([[0, 0]], np.float32),
        fc1T=np.array([[[129]], [[130]]], np.uint8),
        fc2T=np.array([[[129]], [[131]]], np.uint8),
        fc1ScalesT=np.ones((2, 1), np.float32),
        fc2ScalesT=np.ones((2, 1), np.float32),
    )
    assert moe_reference(quantized, dict(expert_weight_bits=8), True)[0, 0] == 6


def test_swiglu_clamps_operands_before_beta_and_product():
    inputs = dict(
        inputT=np.array([[1]], np.float32),
        routerT=np.array([[0]], np.float32),
        fc1T=np.array([[[2], [-3]]], np.float32),
        fc2T=np.array([[[1]]], np.float32),
    )
    attrs = dict(
        activation_type="swiglu",
        swiglu_fusion=1,
        swiglu_limit=0.5,
        activation_alpha=2,
        activation_beta=0.1,
    )
    expected = 0.5 / (1 + np.exp(-1)) * (-0.5 + 0.1)
    np.testing.assert_allclose(moe_reference(inputs, attrs), [[expected]], rtol=1e-7)


def test_linear_per_key_decay_is_applied_before_delta_retrieval():
    inputs = dict(
        queryT=np.array([[[1, 2]]], np.float32),
        keyT=np.array([[[1, 0]]], np.float32),
        valueT=np.array([[[5]]], np.float32),
        pastStateT=np.array([[[[2], [4]]]], np.float32),
        decayT=np.log(np.array([[[0.5, 0.25]]], np.float32)),
        betaT=np.array([[[0.5]]], np.float32),
    )
    output, state = linear_attention_reference(
        inputs, dict(q_num_heads=1, kv_num_heads=1, update_rule="gated_delta", scale=1)
    )
    np.testing.assert_allclose(state, [[[[3], [1]]]], rtol=1e-7)
    np.testing.assert_allclose(output, [[[5]]], rtol=1e-7)


def test_feature_cases_distinguish_capture_prefix_and_first_zero_mask():
    cases = feature_cases()
    capture = next(case for case in cases if "prefix_capture" in case["name"])
    np.testing.assert_array_equal(
        capture["expected"]["stateUpdateT"][1], capture["inputs"]["inputT"][2:5]
    )
    masked = next(case for case in cases if case["name"].endswith("mask_True"))
    np.testing.assert_array_equal(masked["expected"]["maskIndexT"], [1, 0])
    unmasked = next(case for case in cases if case["name"].endswith("mask_False"))
    np.testing.assert_array_equal(unmasked["expected"]["maskIndexT"], [0, 0])


def test_moe_feature_matrix_covers_fusion_quantization_and_scaling():
    cases = advanced_cases()
    assert len(cases) == 22
    dense = [case for case in cases if case["operation"] == "com.microsoft.MoE"]
    quantized = [case for case in cases if case["operation"] == "com.microsoft.QMoE"]
    assert {case["attrs"]["swiglu_fusion"] for case in dense} == {0, 1, 2}
    assert {
        (case["attrs"]["expert_weight_bits"], case["attrs"]["block_size"]) for case in quantized
    } == {(4, 0), (4, 16), (8, 0), (8, 16)}
    assert all(np.isfinite(case["expected"]["outputT"]).all() for case in cases)
