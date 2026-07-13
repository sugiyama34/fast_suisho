"""tools.nnue.leb128 の単体テスト。"""

import numpy as np
import pytest

from tools.nnue import leb128


@pytest.mark.parametrize(
    ("value", "encoded"),
    [
        (0, b"\x00"),
        (1, b"\x01"),
        (-1, b"\x7f"),
        (63, b"\x3f"),
        (64, b"\xc0\x00"),  # 64 は 1 バイトだと bit6 が符号と衝突するため 2 バイト
        (-64, b"\x40"),
        (-65, b"\xbf\x7f"),
        (8191, b"\xff\x3f"),
        (8192, b"\x80\xc0\x00"),
        (-8192, b"\x80\x40"),
        (-8193, b"\xff\xbf\x7f"),
        (32767, b"\xff\xff\x01"),
        (-32768, b"\x80\x80\x7e"),
    ],
)
def test_known_vectors(value: int, encoded: bytes) -> None:
    assert leb128.encode(np.array([value], dtype=np.int16)) == encoded
    assert leb128.decode(encoded, 1, np.int16)[0] == value


def test_int16_exhaustive_roundtrip() -> None:
    values = np.arange(-32768, 32768, dtype=np.int16)
    assert np.array_equal(leb128.decode(leb128.encode(values), len(values), np.int16), values)


def test_empty_roundtrip() -> None:
    empty = np.array([], dtype=np.int16)
    assert leb128.encode(empty) == b""
    assert len(leb128.decode(b"", 0, np.int16)) == 0
    block = leb128.write_block(empty)
    decoded, end = leb128.read_block(block, 0, 0, np.int16)
    assert end == len(block)
    assert len(decoded) == 0


def test_read_block_rejects_truncated_length_field() -> None:
    with pytest.raises(ValueError, match="truncated LEB128 length field"):
        leb128.read_block(leb128.LEB128_MAGIC + b"\x04\x00", 0, 1, np.int16)


def test_read_block_rejects_truncated_body() -> None:
    block = leb128.write_block(np.array([1000], dtype=np.int16))
    with pytest.raises(ValueError, match="truncated LEB128 body"):
        leb128.read_block(block[:-1], 0, 1, np.int16)


def test_decode_accepts_non_minimal_encoding() -> None:
    # 冗長な継続バイト付き 0 (\x80\x00) も C++ reader と同様に受理する
    assert leb128.decode(b"\x80\x00", 1, np.int16)[0] == 0


def test_decode_rejects_wrong_count() -> None:
    with pytest.raises(ValueError, match="expected 3 varints"):
        leb128.decode(b"\x00\x00", 3, np.int16)


def test_decode_rejects_unterminated_varint() -> None:
    with pytest.raises(ValueError, match="trailing bytes"):
        leb128.decode(b"\x00\x80", 1, np.int16)


def test_decode_rejects_overlong_varint() -> None:
    with pytest.raises(ValueError, match="longer than 3 bytes"):
        leb128.decode(b"\x80\x80\x80\x00", 1, np.int16)


def test_block_roundtrip() -> None:
    rng = np.random.default_rng(0)
    values = rng.integers(-32768, 32768, size=10_000, dtype=np.int16)
    block = leb128.write_block(values)
    assert block.startswith(leb128.LEB128_MAGIC)
    decoded, end = leb128.read_block(block, 0, len(values), np.int16)
    assert end == len(block)
    assert np.array_equal(decoded, values)


def test_read_block_rejects_missing_magic() -> None:
    with pytest.raises(ValueError, match="magic not found"):
        leb128.read_block(b"NOT_A_MAGIC" + b"\x00" * 32, 0, 1, np.int16)
