"""nn.bin の parser / serializer。

parse → serialize がバイト一致することを tests/tools/test_roundtrip.py で保証する。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.nnue import leb128
from tools.nnue.model import (
    NNUE_VERSION,
    Arch,
    AffineLayer,
    FeatureTransformer,
    LayerStack,
    NNUEModel,
)


def _read_u32(buf: bytes, offset: int) -> tuple[int, int]:
    field = buf[offset : offset + 4]
    if len(field) != 4:
        raise ValueError(f"truncated u32 field at offset {offset}")
    return int.from_bytes(field, "little"), offset + 4


def _read_affine(buf: bytes, offset: int, out_dims: int, padded_in: int) -> tuple[AffineLayer, int]:
    # frombuffer は bytes 上の read-only view を返すため、編集可能な配列に copy する
    biases = np.frombuffer(buf, dtype="<i4", count=out_dims, offset=offset).copy()
    offset += out_dims * 4
    weights = np.frombuffer(buf, dtype=np.int8, count=out_dims * padded_in, offset=offset).copy()
    offset += out_dims * padded_in
    return AffineLayer(biases=biases, weights=weights.reshape(out_dims, padded_in)), offset


def parse_bytes(buf: bytes, arch: Arch | None = None) -> NNUEModel:
    arch = arch or Arch()
    offset = 0

    version, offset = _read_u32(buf, offset)
    if version != NNUE_VERSION:
        raise ValueError(f"unexpected version 0x{version:08x} (expected 0x{NNUE_VERSION:08x})")
    file_hash, offset = _read_u32(buf, offset)
    arch_len, offset = _read_u32(buf, offset)
    architecture = buf[offset : offset + arch_len]
    if len(architecture) != arch_len:
        raise ValueError("truncated architecture string")
    offset += arch_len

    ft_hash, offset = _read_u32(buf, offset)
    ft_biases, offset = leb128.read_block(buf, offset, arch.half_dims, np.int16)
    ft_weights, offset = leb128.read_block(buf, offset, arch.half_dims * arch.input_dims, np.int16)
    ft = FeatureTransformer(
        hash=ft_hash,
        biases=ft_biases,
        weights=ft_weights.reshape(arch.input_dims, arch.half_dims),
    )

    stacks = []
    for _ in range(arch.layer_stacks):
        stack_hash, offset = _read_u32(buf, offset)
        fc_0, offset = _read_affine(buf, offset, arch.fc0_out, arch.fc0_padded_in)
        fc_1, offset = _read_affine(buf, offset, arch.hidden2_dims, arch.fc1_padded_in)
        fc_2, offset = _read_affine(buf, offset, 1, arch.fc2_padded_in)
        stacks.append(LayerStack(hash=stack_hash, fc_0=fc_0, fc_1=fc_1, fc_2=fc_2))

    if offset != len(buf):
        raise ValueError(f"{len(buf) - offset} trailing bytes after last layer stack")

    return NNUEModel(
        version=version,
        hash=file_hash,
        architecture=architecture,
        feature_transformer=ft,
        layer_stacks=stacks,
        arch=arch,
    )


def serialize_bytes(model: NNUEModel) -> bytes:
    arch = model.arch
    ft = model.feature_transformer
    if ft.biases.shape != (arch.half_dims,) or ft.biases.dtype != np.int16:
        raise ValueError("feature_transformer.biases: expected int16 (half_dims,)")
    if ft.weights.shape != (arch.input_dims, arch.half_dims) or ft.weights.dtype != np.int16:
        raise ValueError("feature_transformer.weights: expected int16 (input_dims, half_dims)")
    if len(model.layer_stacks) != arch.layer_stacks:
        raise ValueError(f"expected {arch.layer_stacks} layer stacks")

    parts = [
        model.version.to_bytes(4, "little"),
        model.hash.to_bytes(4, "little"),
        len(model.architecture).to_bytes(4, "little"),
        model.architecture,
        ft.hash.to_bytes(4, "little"),
        leb128.write_block(ft.biases),
        leb128.write_block(ft.weights),
    ]
    for stack in model.layer_stacks:
        parts.append(stack.hash.to_bytes(4, "little"))
        for layer, out_dims, padded_in in (
            (stack.fc_0, arch.fc0_out, arch.fc0_padded_in),
            (stack.fc_1, arch.hidden2_dims, arch.fc1_padded_in),
            (stack.fc_2, 1, arch.fc2_padded_in),
        ):
            if layer.biases.shape != (out_dims,) or layer.biases.dtype != np.int32:
                raise ValueError(f"affine biases: expected int32 ({out_dims},)")
            if layer.weights.shape != (out_dims, padded_in) or layer.weights.dtype != np.int8:
                raise ValueError(f"affine weights: expected int8 ({out_dims}, {padded_in})")
            parts.append(layer.biases.astype("<i4", copy=False).tobytes())
            parts.append(layer.weights.tobytes())
    return b"".join(parts)


def parse_file(path: str | Path, arch: Arch | None = None) -> NNUEModel:
    return parse_bytes(Path(path).read_bytes(), arch)


def serialize_file(model: NNUEModel, path: str | Path) -> None:
    Path(path).write_bytes(serialize_bytes(model))
