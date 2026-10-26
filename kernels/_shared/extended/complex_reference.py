# SPDX-License-Identifier: Apache-2.0
"""Independent published goldens and NumPy checks for complex Vulkan operators.

References execute only in this development harness. The shipped C++ runtime
never imports Python or computes a tensor result on the host.
"""

import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from kernels import provenance
from pathlib import Path
import subprocess
import time
import numpy as np
from kernels._shared.verify_upstream import materialize, tolerance, compare, UnsupportedFixture

ROOT = Path(__file__).resolve().parents[3]


def typed_descriptors(fixtures, contract, port):
    result = {}
    for name, value in fixtures.items():
        descriptor = dict(value)
        declaration = contract[port][name]
        variable = declaration["dtype"]
        allowed = contract.get("typeConstraints", {}).get(variable, [variable])
        physical = value["dtype"]
        if physical not in allowed:
            storage = declaration.get("storage")
            if storage == physical and len(allowed) == 1:
                descriptor["source_dtype"] = physical
                descriptor["dtype"] = allowed[0]
            else:
                raise UnsupportedFixture(
                    f"Cannot resolve logical dtype of {name}: {physical}, {allowed}, storage={storage}"
                )
        result[name] = descriptor
    return result


def published_cases(full=False):
    names = json.loads((ROOT / "kernels/_shared/extended/complex_ops.json").read_text())
    result = []
    for operation in names:
        document = provenance.document(operation, "tests")
        contract = provenance.document(operation, "contract")
        candidates = []
        for case in document["cases"]:
            if not all(("data" in value for value in case["outputs"].values())):
                continue
            try:
                inputs = {
                    name: materialize(value, document, maximum=8192)
                    for name, value in case["inputs"].items()
                }
                expected = {
                    name: materialize(value, document, maximum=8192)
                    for name, value in case["outputs"].items()
                }
            except (ValueError, UnsupportedFixture):
                continue
            if not full and not any((np.any(value != 0) for value in expected.values())):
                continue
            if not full and operation == "ai.onnx.QLinearMatMul" and inputs["a"].shape[-1] == 0:
                continue
            entry = dict(
                operation=operation,
                name=case["name"],
                inputs=inputs,
                expected=expected,
                attrs=case.get("attrs", {}),
                args=case.get("args", {}),
                comparisons={
                    name: tolerance(value, case) for name, value in case["outputs"].items()
                },
                oracle="published_explicit_output",
                input_descriptors=typed_descriptors(case["inputs"], contract, "inputs"),
                output_descriptors=typed_descriptors(case["outputs"], contract, "outputs"),
            )
            candidates.append(entry)
        candidates.sort(key=lambda row: sum((value.size for value in row["inputs"].values())))
        result += candidates if full else candidates[:1]
    return result


def f(value):
    return np.asarray(value, np.float32)


def softmax(value):
    x = np.asarray(value, np.float64)
    x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return x / np.sum(x, axis=-1, keepdims=True)


def recurrent_linear(query, key, value, initial, decay, beta, scale):
    state = np.asarray(initial, np.float64).copy()
    outputs = []
    for q, k, v, d, b in zip(query, key, value, decay, beta):
        state *= np.exp(d)
        error = v - k @ state
        state += b * np.outer(k, error)
        outputs.append(q @ state * scale)
    return (f(outputs), f(state))


def custom_cases():
    result = []

    def add(operation, inputs, expected, attrs=None):
        result.append(
            dict(
                operation=operation,
                name="independent_nonzero_reference",
                inputs=inputs,
                expected=expected,
                attrs=attrs or {},
                args={},
                comparisons={
                    name: dict(atol=0.0002, rtol=0.0002, allow_nan=False) for name in expected
                },
                oracle="independent_numpy",
            )
        )

    x = f([[[1, 2], [3, -1]]])
    k = f([[[0.5, 0.25], [1, -0.5]]])
    v = f([[[2, 1], [1, 3]]])
    state = f([[[[0.1, -0.2], [0.3, 0.4]]]])
    decay = f([[[np.log(0.9)], [np.log(0.8)]]])
    beta = f([[[0.3], [0.7]]])
    out, final = recurrent_linear(
        x[0], k[0], v[0], state[0, 0], decay.ravel(), beta.ravel(), 1 / np.sqrt(2)
    )
    add(
        "ai.onnx.LinearAttention",
        dict(queryT=x, keyT=k, valueT=v, pastStateT=state, decayT=decay, betaT=beta),
        dict(outputT=out[None], presentStateT=final[None, None]),
        dict(q_num_heads=1, kv_num_heads=1),
    )
    add(
        "com.microsoft.LinearAttention",
        dict(queryT=x, keyT=k, valueT=v, pastStateT=state, decayT=decay, betaT=beta),
        dict(outputT=out[None], presentStateT=final[None, None]),
        dict(q_num_heads=1, kv_num_heads=1),
    )
    out, final = recurrent_linear(x[0], k[0], v[0], state[0, 0].T, decay.ravel(), beta.ravel(), 0.6)
    add(
        "com.microsoft.GatedDeltaNet",
        dict(
            queryT=x.reshape(1, 2, 1, 2),
            keyT=k.reshape(1, 2, 1, 2),
            valueT=v.reshape(1, 2, 1, 2),
            initialStateT=state,
            decayT=decay,
            betaT=beta,
        ),
        dict(outputT=out.reshape(1, 2, 1, 2), finalStateT=final.T[None, None]),
        dict(scale=0.6),
    )
    q = f([[[1, 0.5]]])
    current_k = f([[[0.5, 1]]])
    current_v = f([[[2, 3]]])
    past_k = f([[[[1, 0], [0, 0], [0, 0]]]])
    past_v = f([[[[4, 1], [0, 0], [0, 0]]]])
    keys = np.vstack([past_k[0, 0, :1], current_k[0]])
    vals = np.vstack([past_v[0, 0, :1], current_v[0]])
    scores = q[0].astype(np.float64) @ keys.astype(np.float64).T / np.sqrt(2)
    expected = f(softmax(scores) @ vals)
    present_k = past_k.copy()
    present_k[0, 0, 1] = current_k[0, 0]
    present_v = past_v.copy()
    present_v[0, 0, 1] = current_v[0, 0]
    add(
        "com.microsoft.DecoderMaskedMultiHeadAttention",
        dict(
            queryT=q,
            keyT=current_k,
            valueT=current_v,
            pastKeyT=past_k,
            pastValueT=past_v,
            pastSequenceLengthT=np.array([1], np.int32),
        ),
        dict(outputT=expected[None], presentKeyT=present_k, presentValueT=present_v),
        dict(num_heads=1, past_present_share_buffer=1),
    )
    sparse_k = np.pad(past_k, ((0, 0), (0, 0), (0, 1), (0, 6)))
    sparse_v = np.pad(past_v, ((0, 0), (0, 0), (0, 1), (0, 6)))
    sk = sparse_k.copy()
    sv = sparse_v.copy()
    sk[0, 0, 1, :2] = current_k[0, 0]
    sv[0, 0, 1, :2] = current_v[0, 0]
    add(
        "com.microsoft.SparseAttention",
        dict(
            queryT=np.pad(q, ((0, 0), (0, 0), (0, 6))),
            keyT=np.pad(current_k, ((0, 0), (0, 0), (0, 6))),
            valueT=np.pad(current_v, ((0, 0), (0, 0), (0, 6))),
            pastKeyT=sparse_k,
            pastValueT=sparse_v,
            blockRowIndicesT=np.array([[0, 1]], np.int32),
            blockColIndicesT=np.array([[0]], np.int32),
            totalSequenceLengthT=np.array([2], np.int32),
            keyTotalSequenceLengthsT=np.array([2], np.int32),
        ),
        dict(outputT=np.pad(expected[None], ((0, 0), (0, 0), (0, 6))), pastKeyT=sk, pastValueT=sv),
        dict(num_heads=1, kv_num_heads=1, sparse_block_size=16, scale=1 / np.sqrt(2)),
    )
    add(
        "ai.onnx.CausalConvWithState",
        dict(
            inputT=f([[[1, 2, 3]]]),
            weightT=f([[[1, 2, 3]]]),
            biasT=f([0.1]),
            pastStateT=f([[[4, 5]]]),
        ),
        dict(outputT=f([[[17.1, 13.1, 14.1]]]), presentStateT=f([[[2, 3]]])),
    )
    codebook = f(
        [
            -1,
            -0.6961928009986877,
            -0.5250730514526367,
            -0.3949174880981445,
            -0.28444138169288635,
            -0.18477343022823334,
            -0.09105003625154495,
            0,
            0.07958029955625534,
            0.16093020141124725,
            0.24611230194568634,
            0.33791524171829224,
            0.44070982933044434,
            0.5626170039176941,
            0.7229568362236023,
            1,
        ]
    )
    a = f([[1, -2, 0.5, 3]])
    packed = np.array([240, 120, 105, 171], np.uint8)
    codes = np.asarray([[byte >> 4, byte & 15] for byte in packed]).ravel()
    weights = codebook[codes].reshape(2, 4)
    add(
        "com.microsoft.MatMulBnb4",
        dict(aT=a, bT=packed, absmaxT=f([1])),
        dict(yT=f(a @ weights.T)),
        dict(K=4, N=2, block_size=16, quant_type=1),
    )
    a = f([[1, 2, -1, 0.5]])
    norm = f([1, 0.5, 2, 1])
    skip = f([[0.5, -0.5, 0.25, 1]])
    residual = a + skip
    normalized = (
        residual
        / np.sqrt(np.mean(residual.astype(np.float64) ** 2, axis=-1, keepdims=True) + 1e-06)
        * norm
    )
    inputs = dict(aT=a, skipT=skip, normScaleT=norm)
    expected = dict(residualT=residual)
    for name, byte in (("q", 152), ("k", 168), ("v", 184)):
        inputs[name + "BT"] = np.full((2, 1, 8), byte, np.uint8)
        inputs[name + "ScalesT"] = np.full((2, 1), 0.5, np.float32)
        weights = np.tile(f([0, (byte >> 4) - 8, 0, (byte >> 4) - 8]) * 0.5, (2, 1))
        expected[name + "T"] = f(normalized @ weights.T)
    add(
        "com.microsoft.MatMulNBitsQkv",
        inputs,
        {n: expected[n] for n in ("qT", "kT", "vT", "residualT")},
        dict(K=4, Nq=2, Nkv=2, block_size=16),
    )
    ids = np.array([[0, 1]], np.int32)
    word = f([[1, 2], [4, 8]])
    position = f([[0.5, 0], [1, -1]])
    gamma, bias = (f([1.2, 0.7]), f([-0.2, 0.1]))
    embedding = word[ids] + position[None]
    mean = embedding.astype(np.float64).mean(axis=-1, keepdims=True)
    variance = ((embedding - mean) ** 2).mean(axis=-1, keepdims=True)
    add(
        "com.microsoft.EmbedLayerNormalization",
        dict(
            inputIdsT=ids,
            wordEmbeddingT=word,
            positionEmbeddingT=position,
            gammaT=gamma,
            betaT=bias,
            maskT=np.array([[1, 0]], np.int32),
        ),
        dict(
            outputT=f((embedding - mean) / np.sqrt(variance + 1e-05) * gamma + bias),
            maskIndexT=np.array([1], np.int32),
            embeddingSumT=embedding,
        ),
        dict(epsilon=1e-05),
    )
    query = f([[[1, 2], [-1, 0.5]]])
    qb = f([0.1, -0.2])
    weight = f([[0.5, -0.3], [0.2, 0.7]])
    bias = f([0.2, -0.1])
    eco = f([[[[0.7]]]])
    rel = f([[[[1, 2], [3, 4]]]])
    project = (query.astype(np.float64) + qb) @ weight + bias

    def sig(value):
        return 1 / (1 + np.exp(-value))

    gate = sig(project[..., :1]) * (sig(project[..., 1:]) * 0.7 - 1) + 2
    add(
        "com.microsoft.GatedRelativePositionBias",
        dict(queryLayerT=query, queryBiasT=qb, relPosT=rel, weightT=weight, biasT=bias, ecoAT=eco),
        dict(outputT=f(gate[:, None] * rel)),
        dict(num_heads=1),
    )
    return result


def moe_reference(inputs, attrs, quantized=False):
    """Dense NumPy oracle using public weight packing and routing contracts."""
    x = np.asarray(inputs["inputT"], np.float64)
    hidden = x.shape[-1]
    x = x.reshape(-1, hidden)
    logits = np.asarray(inputs["routerT"], np.float64)
    activation = attrs.get("activation_type", "relu")
    fusion = attrs.get("swiglu_fusion", 0)
    first = np.asarray(inputs["fc1T"])
    second = np.asarray(inputs["fc2T"])
    if quantized:
        bits = attrs.get("expert_weight_bits", 4)
        midpoint = 1 << (bits - 1)
        block = attrs.get("block_size", 0)

        def unpack(packed, scales):
            if bits == 4:
                codes = np.stack((packed & 15, packed >> 4), axis=-1).reshape(
                    *packed.shape[:-1], packed.shape[-1] * 2
                )
            else:
                codes = packed
            positions = (
                np.arange(codes.shape[-1]) // block
                if block
                else np.zeros(codes.shape[-1], np.int64)
            )
            scale = np.asarray(scales, np.float64)
            if not block:
                scale = scale[..., None]
            return (codes.astype(np.float64) - midpoint) * scale[..., positions]

        first = unpack(first, inputs["fc1ScalesT"])
        second = unpack(second, inputs["fc2ScalesT"])
    result = np.zeros_like(x)
    probabilities = softmax(logits)
    for token in range(len(x)):
        chosen = sorted(
            range(logits.shape[1]), key=lambda expert: (logits[token, expert], expert), reverse=True
        )[: attrs.get("k", 1)]
        weights = probabilities[token, chosen]
        if quantized or attrs.get("normalize_routing_weights", 0):
            weights = weights / np.sum(weights)
        for expert, weight in zip(chosen, weights):
            projected = first[expert].astype(np.float64) @ x[token]
            if "fc1BiasT" in inputs:
                projected += inputs["fc1BiasT"][expert]
            if activation == "swiglu" and fusion:
                gate, linear = (
                    (projected[::2], projected[1::2]) if fusion == 1 else np.split(projected, 2)
                )
            else:
                gate = projected
                linear = None
                if "fc3T" in inputs:
                    linear = np.asarray(inputs["fc3T"][expert], np.float64) @ x[token]
                    if "fc3BiasT" in inputs:
                        linear += inputs["fc3BiasT"][expert]
            if activation == "swiglu":
                if "swiglu_limit" in attrs:
                    limit = attrs["swiglu_limit"]
                    gate = np.minimum(gate, limit)
                    linear = np.clip(linear, -limit, limit)
                transformed = (
                    gate
                    / (1 + np.exp(-attrs.get("activation_alpha", 1) * gate))
                    * (linear + attrs.get("activation_beta", 0))
                )
            elif activation == "silu":
                transformed = gate / (1 + np.exp(-gate))
                if linear is not None:
                    transformed *= linear
            elif activation == "gelu":
                import math

                transformed = gate * 0.5 * (1 + np.vectorize(math.erf)(gate / np.sqrt(2)))
            elif activation == "relu":
                transformed = np.maximum(gate, 0)
            else:
                transformed = gate
            projected = second[expert].astype(np.float64) @ transformed
            if "fc2BiasT" in inputs:
                projected += inputs["fc2BiasT"][expert]
            result[token] += weight * projected
    return f(result.reshape(inputs["inputT"].shape))


def advanced_cases():
    """Deterministic independent cases cover supported fused feature combinations."""
    rng = np.random.default_rng(20261026)
    result = []

    def add(operation, name, inputs, expected, attrs):
        result.append(
            dict(
                operation=operation,
                name=name,
                inputs=inputs,
                expected=expected,
                attrs=attrs,
                args={},
                comparisons={n: dict(atol=2e-5, rtol=2e-5, allow_nan=False) for n in expected},
                oracle="independent_numpy_seed_20261026",
            )
        )

    for activation, fusion, limit in [
        ("relu", 0, None),
        ("gelu", 0, None),
        ("identity", 0, None),
        ("silu", 0, None),
        ("swiglu", 0, None),
        ("swiglu", 1, None),
        ("swiglu", 2, 0.35),
    ]:
        for normalize in [0, 1]:
            hidden, intermediate, experts = 4, 3, 3
            rows = intermediate * (2 if activation == "swiglu" and fusion else 1)
            inputs = dict(
                inputT=f(rng.normal(size=(2, 4))),
                routerT=f([[0.7, 0.7, -0.2], [-0.3, 0.9, 0.8]]),
                fc1T=f(rng.normal(size=(experts, rows, hidden)) * 0.4),
                fc2T=f(rng.normal(size=(experts, hidden, intermediate)) * 0.3),
                fc1BiasT=f(rng.normal(size=(experts, rows)) * 0.1),
                fc2BiasT=f(rng.normal(size=(experts, hidden)) * 0.1),
            )
            if activation in ("silu", "swiglu") and fusion == 0:
                inputs.update(
                    fc3T=f(rng.normal(size=(experts, intermediate, hidden)) * 0.4),
                    fc3BiasT=f(rng.normal(size=(experts, intermediate)) * 0.1),
                )
            attrs = dict(
                activation_type=activation,
                swiglu_fusion=fusion,
                k=2,
                normalize_routing_weights=normalize,
                activation_alpha=1.702,
                activation_beta=0.05,
            )
            if limit is not None:
                attrs["swiglu_limit"] = limit
            add(
                "com.microsoft.MoE",
                f"independent_{activation}_fusion{fusion}_normalize{normalize}",
                inputs,
                dict(outputT=moe_reference(inputs, attrs)),
                attrs,
            )
    for bits in [4, 8]:
        for block in [0, 16]:
            for activation in ["relu", "swiglu"]:
                hidden, intermediate, experts = (4, 4, 2) if block == 0 else (16, 16, 2)
                rows = intermediate * (2 if activation == "swiglu" else 1)
                inputs = dict(
                    inputT=f(rng.normal(size=(2, hidden)) * 0.2),
                    routerT=f([[0.1, 0.1], [0.2, -0.1]]),
                    fc1T=rng.integers(0, 256, (experts, rows, hidden * bits // 8), dtype=np.uint8),
                    fc2T=rng.integers(
                        0, 256, (experts, hidden, intermediate * bits // 8), dtype=np.uint8
                    ),
                )
                inputs["fc1ScalesT"] = f(
                    rng.uniform(
                        0.005, 0.02, (experts, rows, hidden // block) if block else (experts, rows)
                    )
                )
                inputs["fc2ScalesT"] = f(
                    rng.uniform(
                        0.005,
                        0.02,
                        (experts, hidden, intermediate // block) if block else (experts, hidden),
                    )
                )
                attrs = dict(
                    expert_weight_bits=bits,
                    block_size=block,
                    activation_type=activation,
                    swiglu_fusion=1 if activation == "swiglu" else 0,
                    k=2,
                    activation_alpha=0.9,
                    activation_beta=-0.02,
                    weights_prepacked=0,
                )
                if activation == "swiglu":
                    attrs["swiglu_limit"] = 0.75
                add(
                    "com.microsoft.QMoE",
                    f"independent_q{bits}_block{block}_{activation}",
                    inputs,
                    dict(outputT=moe_reference(inputs, attrs, True)),
                    attrs,
                )
    return result


def linear_attention_reference(inputs, attrs):
    q, key, value = (np.asarray(inputs[n], np.float64) for n in ("queryT", "keyT", "valueT"))
    batch, sequence, _ = q.shape
    qh, kh = attrs["q_num_heads"], attrs["kv_num_heads"]
    dk, dv = q.shape[-1] // qh, value.shape[-1] // kh
    keyheads = key.shape[-1] // dk
    q = q.reshape(batch, sequence, qh, dk)
    key = key.reshape(batch, sequence, keyheads, dk)
    value = value.reshape(batch, sequence, kh, dv)
    incoming = np.asarray(inputs.get("pastStateT", np.zeros((batch, kh, dk, dv))), np.float64)
    window = attrs.get("state_window", 0)
    state = incoming[-1].copy() if window else incoming.copy()
    history = list(incoming) if window else []
    outputs = np.zeros((batch, sequence, max(qh, kh), dv))
    scale = attrs.get("scale", 0) or 1 / np.sqrt(dk)
    rule = attrs.get("update_rule", "gated_delta")
    for t in range(sequence):
        for b in range(batch):
            for h in range(kh):
                k = key[b, t, h // (kh // keyheads)]
                v = value[b, t, h]
                if rule in ("gated", "gated_delta"):
                    decay = inputs["decayT"][b, t]
                    decay = decay[h * dk : (h + 1) * dk] if len(decay) == kh * dk else decay[h]
                    state[b, h] *= np.exp(np.asarray(decay).reshape(-1, 1))
                update = v
                if rule in ("delta", "gated_delta"):
                    beta = inputs["betaT"][b, t]
                    update = beta[0 if len(beta) == 1 else h] * (v - k @ state[b, h])
                state[b, h] += np.outer(k, update)
            for h in range(max(qh, kh)):
                outputs[b, t, h] = (
                    q[b, t, h // (max(qh, kh) // qh)] @ state[b, h // (max(qh, kh) // kh)] * scale
                )
        if window:
            history.append(state.copy())
    return f(outputs.reshape(batch, sequence, -1)), f(
        np.asarray(history[-window:]) if window else state
    )


def feature_cases():
    rng = np.random.default_rng(20261027)
    result = []

    def add(operation, name, inputs, expected, attrs):
        result.append(
            dict(
                operation=operation,
                name=name,
                inputs=inputs,
                expected=expected,
                attrs=attrs,
                args={},
                comparisons={n: dict(atol=2e-5, rtol=2e-5, allow_nan=False) for n in expected},
                oracle="independent_numpy_feature_reference",
            )
        )

    for operation in ["ai.onnx.LinearAttention", "com.microsoft.LinearAttention"]:
        for rule in ["linear", "gated", "delta", "gated_delta"]:
            inverse = operation.startswith("com.microsoft")
            qh, kh, dk, dv = (1, 2, 3, 2) if inverse else (2, 1, 3, 2)
            keyheads = 1
            inputs = dict(
                queryT=f(rng.normal(size=(1, 3, qh * dk)) * 0.3),
                keyT=f(rng.normal(size=(1, 3, keyheads * dk)) * 0.3),
                valueT=f(rng.normal(size=(1, 3, kh * dv))),
                pastStateT=f(rng.normal(size=(1, kh, dk, dv)) * 0.1),
            )
            if rule in ("gated", "gated_delta"):
                inputs["decayT"] = f(rng.uniform(-0.5, -0.01, (1, 3, kh * dk)))
            if rule in ("delta", "gated_delta"):
                inputs["betaT"] = f(rng.uniform(0.1, 0.9, (1, 3, kh)))
            attrs = dict(q_num_heads=qh, kv_num_heads=kh, update_rule=rule, scale=0.6)
            out, state = linear_attention_reference(inputs, attrs)
            add(
                operation,
                f"independent_{rule}_per_key_decay_shared_keys",
                inputs,
                dict(outputT=out, presentStateT=state),
                attrs,
            )
    inputs = dict(
        queryT=f(rng.normal(size=(1, 1, 3)) * 0.2),
        keyT=f(rng.normal(size=(1, 1, 3)) * 0.3),
        valueT=f(rng.normal(size=(1, 1, 2))),
        pastStateT=f(rng.normal(size=(3, 1, 1, 3, 2)) * 0.1),
    )
    attrs = dict(q_num_heads=1, kv_num_heads=1, update_rule="linear", state_window=3)
    out, state = linear_attention_reference(inputs, attrs)
    add(
        "com.microsoft.LinearAttention",
        "independent_rollback_short_prefix",
        inputs,
        dict(outputT=out, presentStateT=state),
        attrs,
    )
    x = f([[[1, 2], [3, 4], [5, 6]]])
    past = f(np.arange(-8, 0).reshape(1, 4, 2))
    weights = f([[[1, 2, 3]], [[-1, 0.5, 2]]])
    incoming = np.stack([past - 10, past - 5, past])
    raw = np.concatenate([past[0], x[0]], axis=0)
    output = np.empty_like(x)
    for t in range(3):
        for c in range(2):
            output[0, t, c] = sum(raw[t + 2 * tap, c] * weights[c, 0, tap] for tap in range(3))
    states = f(np.stack([raw[t : t + 4] for t in [1, 2, 3]])[:, None])
    add(
        "com.microsoft.CausalConvWithState",
        "independent_channels_last_dilated_rollback",
        dict(inputT=x, weightT=weights, pastStateT=incoming),
        dict(outputT=output, presentStateT=states),
        dict(channels_last=1, dilation=2, state_window=3),
    )
    x = f(np.arange(12).reshape(6, 2) / 3)
    past = f(np.arange(-8, 0).reshape(2, 2, 2) / 3)
    weights = f([[[1, 2, -1]], [[0.5, -2, 3]]])
    cu = np.array([0, 2, 6], np.int32)
    out = np.zeros_like(x)
    final = np.zeros_like(past)
    capture = np.zeros((2, 3, 2), np.float32)
    for b in range(2):
        current = x[cu[b] : cu[b + 1]]
        raw = np.concatenate([past[b].T, current], axis=0)
        for t in range(len(current)):
            out[cu[b] + t] = np.sum(raw[t : t + 3] * weights[:, 0].T, axis=0)
        final[b] = raw[-2:].T
        count = min([1, 5][b], len(current), 3)
        capture[b, :count] = current[:count]
    add(
        "com.microsoft.VarlenCausalConvWithState",
        "independent_prefix_capture_and_boundaries",
        dict(
            inputT=x,
            weightT=weights,
            cumulativeSequenceLengthT=cu,
            initialStateT=past,
            captureCountT=np.array([1, 5], np.int32),
        ),
        dict(outputT=out, finalStateT=final, stateUpdateT=capture),
        dict(state_update_capacity=3),
    )
    ids = np.array([[0, 1, 0], [1, 0, 1]], np.int32)
    word = np.array([[1000, 0.1, -2], [2, -1, 0.3]], np.float16)
    segment = np.array([[-999, 0.2, 3]], np.float16)
    position = np.array([[0.12, 0.3, -0.5], [0.2, 0.1, 0.4], [-0.2, 0.5, 0.1]], np.float16)
    gamma, bias = (np.array([1.2, 0.8, 0.7], np.float16), np.array([0.1, -0.1, 0.2], np.float16))
    for masked in [False, True]:
        embedded = (
            (word[ids].astype(np.float32) + segment.astype(np.float32))
            .astype(np.float16)
            .astype(np.float32)
            + position[None].astype(np.float32)
        ).astype(np.float16)
        mean = embedded.astype(np.float64).mean(-1, keepdims=True)
        var = ((embedded.astype(np.float64) - mean) ** 2).mean(-1, keepdims=True)
        inputs = dict(
            inputIdsT=ids,
            wordEmbeddingT=word,
            segmentEmbeddingT=segment,
            positionEmbeddingT=position,
            positionIdsT=np.array([[0, 1, 2]], np.int32),
            gammaT=gamma,
            betaT=bias,
        )
        if masked:
            inputs["maskT"] = np.array([[1, 0, 1], [0, 1, 1]], np.int32)
        expected = dict(
            outputT=((embedded - mean) / np.sqrt(var + 1e-5) * gamma + bias).astype(np.float16),
            embeddingSumT=embedded,
            maskIndexT=np.array([1, 0] if masked else [0, 0], np.int32),
        )
        add(
            "com.microsoft.EmbedLayerNormalization",
            f"independent_f16_staging_first_zero_mask_{masked}",
            inputs,
            expected,
            dict(epsilon=1e-5),
        )
    q = f([[[1, 0.5]]])
    k = f([[[[1, 0], [0.5, 1], [-1, 0.5]]]])
    v = f([[[[4, 1], [2, 3], [-2, 1]]]])
    scores = q[0].astype(np.float64) @ k[0, 0].astype(np.float64).T / np.sqrt(2) + np.array(
        [[0, -1, -1]]
    )
    add(
        "com.microsoft.DecoderMaskedMultiHeadAttention",
        "independent_cross_additive_mask_qk",
        dict(queryT=q, keyT=k, valueT=v, maskIndexT=np.array([[1, 0, 0]], np.int32)),
        dict(outputT=f((softmax(scores) @ v[0, 0])[None]), qkT=f(scores.reshape(1, 1, 1, 3))),
        dict(num_heads=1, output_qk=1, mask_filter_value=-1),
    )
    return result


def representatives():
    cases = {case["operation"]: case for case in published_cases()}
    for case in custom_cases():
        cases[case["operation"]] = case
    names = json.loads((ROOT / "kernels/_shared/extended/complex_ops.json").read_text())
    missing = set(names) - set(cases)
    if missing:
        raise RuntimeError(
            "Complex numerical representatives missing: " + ", ".join(sorted(missing))
        )
    return [cases[name] for name in names]


def selection_audit(selected=None):
    eligible = {(row["operation"], row["name"]) for row in published_cases(full=True)}
    omissions = []
    total = 0
    names = json.loads((ROOT / "kernels/_shared/extended/complex_ops.json").read_text())
    for operation in names:
        if selected and operation not in selected:
            continue
        document = provenance.document(operation, "tests")
        for case in document["cases"]:
            total += 1
            if (operation, case["name"]) in eligible:
                continue
            reason = "No published explicit output and no general independent oracle for this frozen fixture"
            if all("data" in value for value in case["outputs"].values()):
                try:
                    for value in list(case["inputs"].values()) + list(case["outputs"].values()):
                        materialize(value, document, maximum=8192)
                except (ValueError, UnsupportedFixture) as error:
                    reason = str(error)
            omissions.append(dict(operation=operation, case=case["name"], reason=reason))
    return dict(
        total_frozen_cases=total,
        eligible_explicit_cases=total - len(omissions),
        unsupported=len(omissions),
        omissions=omissions,
    )


def execute_case(dev, root, number, case):
    path = root / f"case-{number:04d}"
    path.mkdir()
    job = dict(
        operation=case["operation"],
        attrs=case["attrs"],
        args=case["args"],
        package=dev["package"],
        inputs=[],
        outputs=[],
        output_directory=str(path / "actual"),
    )
    for name, array in case["inputs"].items():
        data = path / (name + ".bin")
        data.write_bytes(array.tobytes())
        descriptor = case.get("input_descriptors", {}).get(name, {})
        logical = descriptor.get("dtype", str(array.dtype))
        job["inputs"].append(
            dict(
                name=name,
                dtype=logical,
                storage_dtype=str(array.dtype),
                shape=list(array.shape),
                file=str(data),
            )
        )
    for name, expected in case["expected"].items():
        descriptor = case.get("output_descriptors", {}).get(name, {})
        job["outputs"].append(
            dict(
                name=name,
                dtype=descriptor.get("dtype", str(expected.dtype)),
                storage_dtype=str(expected.dtype),
                shape=list(expected.shape),
            )
        )
    (path / "job.json").write_text(json.dumps(job))
    row = dict(
        operation=case["operation"],
        case=case["name"],
        status="failed",
        oracle=case["oracle"],
        case_directory=str(path),
    )
    started = time.monotonic()
    try:
        proc = subprocess.run(
            [dev["runner"], str(path / "job.json")], capture_output=True, text=True, timeout=300
        )
        if proc.returncode:
            raise RuntimeError(proc.stderr.strip())
        result = json.loads(proc.stdout)
        row["evidence"] = result["info"]
        if (
            result["info"].get("backend") != "software_vulkan"
            or result["info"].get("cpu_compute_fallback") is not False
        ):
            raise RuntimeError(
                "Independent test requires explicitly identified software Vulkan, no CPU numerical fallback"
            )
        checks = {}
        for descriptor in result["outputs"]:
            name = descriptor["name"]
            expected = case["expected"][name]
            actual = np.frombuffer(
                Path(descriptor["file"]).read_bytes(),
                dtype=descriptor.get("storage_dtype", descriptor["dtype"]),
            ).reshape(descriptor["shape"])
            checks[name] = compare(actual, expected, **case["comparisons"][name])
            (path / (name + ".expected.bin")).write_bytes(expected.tobytes())
        row["comparisons"] = checks
        row["status"] = (
            "passed"
            if len(checks) == len(case["expected"]) and all((c["passed"] for c in checks.values()))
            else "failed"
        )
    except Exception as error:
        row["error"] = str(error)
        row["failure_class"] = (
            "process_timeout"
            if isinstance(error, subprocess.TimeoutExpired)
            else "execution_or_contract_error"
        )
    finally:
        row["process_wall_seconds"] = time.monotonic() - started
    return row


def run(development, output, *, full=False, selected=None, jobs=1, case_names=None):
    if not 1 <= jobs <= 8:
        raise ValueError("Conformance jobs must be 1 through 8")
    dev = json.loads(Path(development).read_text())
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    rows = []
    audit = selection_audit(selected)
    cases = (
        published_cases(full=True) + custom_cases() + advanced_cases() + feature_cases()
        if full
        else representatives()
    )
    if selected:
        cases = [case for case in cases if case["operation"] in selected]

    if case_names:
        cases = [case for case in cases if case["name"] in case_names]
    if not cases:
        raise ValueError("No matching independent native cases")

    def tasks():
        if jobs == 1:
            for number, case in enumerate(cases):
                yield execute_case(dev, root, number, case)
        else:
            with ThreadPoolExecutor(max_workers=jobs) as pool:
                pending = [
                    pool.submit(execute_case, dev, root, number, case)
                    for number, case in enumerate(cases)
                ]
                for task in as_completed(pending):
                    yield task.result()

    for row in tasks():
        rows.append(row)
        report = dict(
            kind="xrgpu.complex-independent-software-vulkan",
            physical_qualification=False,
            process_timeout_seconds=300,
            planned_case_count=len(cases),
            executed_case_count=len(rows),
            complete_published_coverage=full and not audit["unsupported"],
            selection=audit,
            harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            source_sha256=dev["source_sha256"],
            runner_sha256=dev["runner_sha256"],
            passed=sum((row["status"] == "passed" for row in rows)),
            failed=sum((row["status"] != "passed" for row in rows)),
            results=rows,
        )
        (root / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-explicit", action="store_true")
    parser.add_argument("--operator", action="append")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--case", action="append")
    args = parser.parse_args()
    result = run(
        args.development,
        args.output,
        full=args.full_explicit,
        selected=args.operator,
        jobs=args.jobs,
        case_names=args.case,
    )
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "failed": result["failed"],
                "planned": result["planned_case_count"],
                "unsupported_frozen": result["selection"]["unsupported"],
                "physical_qualification": False,
            }
        )
    )
    return bool(result["failed"])


if __name__ == "__main__":
    raise SystemExit(main())
