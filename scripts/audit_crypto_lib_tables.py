#!/usr/bin/env python3
"""Independent audit for scripts/crypto_lib.py constant-table helpers.

Critical tables are derived from their definitions rather than retyped:
- MD5 T[i] = floor(abs(sin(i)) * 2^32), RFC 1321.
- SHA-256 init/K = first 32 bits of fractional sqrt/cbrt of primes, FIPS 180-4.
- AES S-box = GF(2^8) inverse modulo x^8+x^4+x^3+x+1, then affine map, FIPS 197.

DES/Blowfish/Base64/TEA/CRC32 are checked against short literal references
because their publication definitions are tables/alphabet constants rather than
cheap arithmetic derivations.
"""
from __future__ import annotations

import math
import sys
from decimal import Decimal, getcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crypto_lib as c  # noqa: E402

getcontext().prec = 80
TWO32 = 1 << 32


def _frac32(x: Decimal) -> int:
    return int((x - int(x)) * TWO32)


def _cbrt(n: int) -> Decimal:
    x = Decimal(n)
    guess = Decimal(str(n ** (1 / 3)))
    for _ in range(30):
        guess = (2 * guess + x / (guess * guess)) / 3
    return guess


def _primes(n: int) -> list[int]:
    out = []
    candidate = 2
    while len(out) < n:
        for p in out:
            if p * p > candidate:
                break
            if candidate % p == 0:
                break
        else:
            out.append(candidate)
            candidate += 1
            continue
        if any(candidate % p == 0 for p in out if p * p <= candidate):
            candidate += 1
            continue
        out.append(candidate)
        candidate += 1
    return out


def _md5_t(n: int = 16) -> list[int]:
    return [int(abs(math.sin(i)) * TWO32) & 0xFFFFFFFF for i in range(1, n + 1)]


def _sha256_init() -> list[int]:
    return [_frac32(Decimal(p).sqrt()) for p in _primes(8)]


def _sha256_k(n: int = 8) -> list[int]:
    return [_frac32(_cbrt(p)) for p in _primes(n)]


def _gf_mul(a: int, b: int) -> int:
    out = 0
    for _ in range(8):
        if b & 1:
            out ^= a
        carry = a & 0x80
        a = (a << 1) & 0xFF
        if carry:
            a ^= 0x1B
        b >>= 1
    return out


def _gf_pow(a: int, e: int) -> int:
    out = 1
    while e:
        if e & 1:
            out = _gf_mul(out, a)
        a = _gf_mul(a, a)
        e >>= 1
    return out


def _rotl8(x: int, n: int) -> int:
    return ((x << n) | (x >> (8 - n))) & 0xFF


def _aes_sbox() -> list[int]:
    vals = []
    for x in range(256):
        inv = 0 if x == 0 else _gf_pow(x, 254)
        vals.append(inv ^ _rotl8(inv, 1) ^ _rotl8(inv, 2) ^ _rotl8(inv, 3) ^ _rotl8(inv, 4) ^ 0x63)
    return vals


LITERAL_EXPECTED = {
    "md5_init": [0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476],
    "sha1_init": [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0],
    "aes_rcon": [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80],
    "chacha20_init": [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
    "salsa20_init": [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
    "tea_delta": 0x9E3779B9,
    "xtea_delta": 0x9E3779B9,
    "blowfish_p_array": [0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344, 0xA4093822, 0x299F31D0, 0x082EFA98, 0xEC4E6C89, 0x452821E6, 0x38D01377, 0xBE5466CF, 0x34E90C6C, 0xC0AC29B7, 0xC97C50DD, 0x3F84D5B5, 0xB5470917, 0x9216D5D9, 0x8979FB1B],
    "crc32_poly": 0xEDB88320,
    "des_ip_table": [58, 50, 42, 34, 26, 18, 10, 2, 60, 52, 44, 36, 28, 20, 12, 4, 62, 54, 46, 38, 30, 22, 14, 6, 64, 56, 48, 40, 32, 24, 16, 8, 57, 49, 41, 33, 25, 17, 9, 1, 59, 51, 43, 35, 27, 19, 11, 3, 61, 53, 45, 37, 29, 21, 13, 5, 63, 55, 47, 39, 31, 23, 15, 7],
    "des_fp_table": [40, 8, 48, 16, 56, 24, 64, 32, 39, 7, 47, 15, 55, 23, 63, 31, 38, 6, 46, 14, 54, 22, 62, 30, 37, 5, 45, 13, 53, 21, 61, 29, 36, 4, 44, 12, 52, 20, 60, 28, 35, 3, 43, 11, 51, 19, 59, 27, 34, 2, 42, 10, 50, 18, 58, 26, 33, 1, 41, 9, 49, 17, 57, 25],
    "base64_alphabet": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/",
}


def _assert_equal(name: str, got, want) -> None:
    if got != want:
        raise AssertionError(f"{name} mismatch\n got={got!r}\nwant={want!r}")


def main() -> None:
    _assert_equal("md5_sine_table", c.md5_sine_table(), _md5_t(16))
    _assert_equal("sha256_init", c.sha256_init(), _sha256_init())
    _assert_equal("sha256_K", c.sha256_K(), _sha256_k(8))
    _assert_equal("aes_forward_sbox", c.aes_forward_sbox(), _aes_sbox())

    for name, want in LITERAL_EXPECTED.items():
        _assert_equal(name, getattr(c, name)(), want)

    boxes = c.des_sboxes()
    assert len(boxes) == 8 and all(len(box) == 4 and all(len(row) == 16 for row in box) for box in boxes), "DES S-box shape wrong"
    assert sorted(c.aes_forward_sbox()) == list(range(256)), "AES S-box is not a byte permutation"
    print("crypto_lib derived table audit PASS")


if __name__ == "__main__":
    main()
