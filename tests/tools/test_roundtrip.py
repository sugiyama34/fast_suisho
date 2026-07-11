"""nn.bin parser / serializer の往復 (parse → serialize) バイト一致テスト。

実ファイルのテストは ../suisho11/nn.bin が存在し、かつ sha256 が
experiments/000-baseline/report.md の fingerprint と一致する場合のみ実行する。
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from tools.nnue import Arch, NNUE_VERSION, io
from tools.nnue.model import AffineLayer, FeatureTransformer, LayerStack, NNUEModel

REPO_ROOT = Path(__file__).resolve().parents[2]
NN_BIN = REPO_ROOT.parent / "suisho11" / "nn.bin"
NN_BIN_SHA256 = "a78b7f889843037d344f482623b3febd124ead5c1f34f134d9f1c2c78cd0f829"

TINY_ARCH = Arch(input_dims=50, half_dims=16, layer_stacks=2)


def _random_model(arch: Arch, seed: int = 0) -> NNUEModel:
    rng = np.random.default_rng(seed)

    def affine(out_dims: int, padded_in: int) -> AffineLayer:
        return AffineLayer(
            biases=rng.integers(-(2**31), 2**31, size=out_dims, dtype=np.int32),
            weights=rng.integers(-128, 128, size=(out_dims, padded_in), dtype=np.int8),
        )

    ft = FeatureTransformer(
        hash=0x5F1348B8,
        biases=rng.integers(-32768, 32768, size=arch.half_dims, dtype=np.int16),
        weights=rng.integers(-32768, 32768, size=(arch.input_dims, arch.half_dims), dtype=np.int16),
    )
    stacks = [
        LayerStack(
            hash=0x6333718A,
            fc_0=affine(arch.fc0_out, arch.fc0_padded_in),
            fc_1=affine(arch.hidden2_dims, arch.fc1_padded_in),
            fc_2=affine(1, arch.fc2_padded_in),
        )
        for _ in range(arch.layer_stacks)
    ]
    return NNUEModel(
        version=NNUE_VERSION,
        hash=0x3C203E1C,
        architecture=b"TestArch",
        feature_transformer=ft,
        layer_stacks=stacks,
        arch=arch,
    )


def test_synthetic_roundtrip() -> None:
    model = _random_model(TINY_ARCH)
    buf = io.serialize_bytes(model)
    parsed = io.parse_bytes(buf, TINY_ARCH)

    assert parsed.version == model.version
    assert parsed.hash == model.hash
    assert parsed.architecture == model.architecture
    assert parsed.feature_transformer.hash == model.feature_transformer.hash
    assert np.array_equal(parsed.feature_transformer.biases, model.feature_transformer.biases)
    assert np.array_equal(parsed.feature_transformer.weights, model.feature_transformer.weights)
    for got, want in zip(parsed.layer_stacks, model.layer_stacks, strict=True):
        assert got.hash == want.hash
        for name in ("fc_0", "fc_1", "fc_2"):
            assert np.array_equal(getattr(got, name).biases, getattr(want, name).biases)
            assert np.array_equal(getattr(got, name).weights, getattr(want, name).weights)

    assert io.serialize_bytes(parsed) == buf


def test_parse_rejects_trailing_bytes() -> None:
    buf = io.serialize_bytes(_random_model(TINY_ARCH)) + b"\x00"
    with pytest.raises(ValueError, match="trailing bytes"):
        io.parse_bytes(buf, TINY_ARCH)


def test_parse_rejects_bad_version() -> None:
    buf = bytearray(io.serialize_bytes(_random_model(TINY_ARCH)))
    buf[0] ^= 0xFF
    with pytest.raises(ValueError, match="unexpected version"):
        io.parse_bytes(bytes(buf), TINY_ARCH)


def _real_nn_bin() -> bytes:
    if not NN_BIN.exists():
        pytest.skip(f"{NN_BIN} not found")
    buf = NN_BIN.read_bytes()
    if hashlib.sha256(buf).hexdigest() != NN_BIN_SHA256:
        pytest.skip("nn.bin does not match the baseline fingerprint")
    return buf


def test_real_nn_bin_roundtrip_byte_exact() -> None:
    buf = _real_nn_bin()
    model = io.parse_bytes(buf)

    assert model.architecture.startswith(b"ModelType=SFNNWithoutPsqt;")
    assert model.hash == 0x3C203E1C
    assert model.feature_transformer.hash == 0x5F1348B8
    assert len(model.layer_stacks) == 9
    assert len({stack.hash for stack in model.layer_stacks}) == 1

    assert io.serialize_bytes(model) == buf
