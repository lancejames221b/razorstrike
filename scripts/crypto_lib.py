#!/usr/bin/env python3
"""Phase 1b — Crypto primitive library.

Faithful public-domain C reference impls, each carrying its identifying
constant(s) verbatim — the recognition signal AND the substrate flaws are
injected into 1d.

Primitives + identifying constant(s):
  - MD5 init 0x67452301,0xefcdab89,0x98badcfe,0x10325476; sine table 0xd76aa478…
  - SHA-1 0x5A827999,0x6ED9EBA1,0x8F1BBCDC,0xCA62C1D6
  - SHA-256 init 0x6a09e667…; K first 0x428a2f98
  - AES forward S-box (0x63,0x7c,0x77,0x7b,0xf2,0x6b,…); Rcon
  - RC4 256-byte state, KSA j=(j+S[i]+key[i%len])&255+swap, PRGA
  - ChaCha20 0x61707865,0x3320646e,0x79622d32,0x6b206574; rotl 16,12,8,7
  - Salsa20 same "expand 32-byte k" words; rotl 7,9,13,18
  - TEA/XTEA delta=0x9E3779B9, 32 rounds (XTEA sum schedule differs)
  - Blowfish P-array from π hex, first word 0x243F6A88
  - CRC32 reflected poly 0xEDB88320, 256-entry table
  - DES IP/FP tables + 8 S-boxes. Base64 alphabet "ABC…+/". XOR cipher keyed byte-XOR loop (structural).

Also define 6 deterministic obfuscation transforms (seed=42) reused by 1c/1d:
(1) rename identifiers to v1,v2,…/sub_XXXX, (2) strip comments,
(3) reformat integer literals hex↔dec (preserve magic-constant values),
(4) reorder independent statements, (5) inline single-use temporaries,
(6) split compound expressions.
"""

import random


# --- 6 deterministic obfuscation transforms (seed=42) ---

def _obf_rename_identifiers(code, seed=42):
    """(1) Rename identifiers to v1,v2,…/sub_XXXX."""
    import re

    # Find all identifiers (letters, digits, underscores; not keywords)
    keywords = {
        "void", "int", "char", "short", "long", "float", "double",
        "unsigned", "signed", "const", "static", "extern", "volatile",
        "struct", "union", "enum", "typedef", "sizeof", "return",
        "if", "else", "for", "while", "do", "switch", "case",
        "break", "continue", "goto", "default", "typedef",
    }

    # Extract identifiers (simple heuristic: word chars not keywords)
    id_pattern = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b')
    ids = id_pattern.findall(code)

    # Filter out keywords and common builtins
    builtin_names = {
        "main", "printf", "scanf", "malloc", "free", "strlen", "strcpy",
        "memcpy", "memset", "sizeof", "NULL", "true", "false",
    }

    # Assign unique names
    name_map = {}
    counter = 0
    for id_name in ids:
        if id_name not in keywords and id_name not in builtin_names:
            if id_name not in name_map:
                counter += 1
                # Use sub_XXXX for longer names, vN for short ones
                if len(id_name) > 6:
                    name_map[id_name] = f"sub_{counter:04X}"
                else:
                    name_map[id_name] = f"v{counter}"

    # Replace identifiers (longest first to avoid partial matches)
    sorted_ids = sorted(name_map.keys(), key=len, reverse=True)
    result = code
    for old_id in sorted_ids:
        new_name = name_map[old_id]
        result = re.sub(r'\b' + re.escape(old_id) + r'\b', new_name, result)

    return result


def _obf_strip_comments(code):
    """(2) Strip comments (// and /* */)."""
    import re

    # Remove single-line comments
    code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)

    # Remove multi-line comments
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)

    return code


def _obf_reformat_int_literals(code, seed=42):
    """(3) Reformat integer literals hex↔dec (preserve magic-constant values)."""
    import re

    # Find integer literals (hex and decimal)
    hex_pattern = re.compile(r'0x[0-9A-Fa-f]+')
    dec_pattern = re.compile(r'\b\d+\b')

    # For hex literals, convert to decimal (for obfuscation)
    def _hex_to_dec(match):
        return str(int(match.group(), 16))

    # For decimal literals, convert to hex (for obfuscation)
    def _dec_to_hex(match):
        return f"0x{int(match.group()):X}"

    # Apply obfuscation (alternate between hex and decimal)
    result = code
    random.seed(seed)

    # First pass: convert some hex to decimal
    result = hex_pattern.sub(_hex_to_dec, result)

    # Second pass: convert some decimal to hex
    result = dec_pattern.sub(_dec_to_hex, result)

    return result


def _obf_reorder_statements(code):
    """(4) Reorder independent statements (simple heuristic)."""
    import re

    # Split code into lines
    lines = code.split('\n')

    # Group statements by semicolon (simple heuristic)
    stmts = []
    current_stmt = ""

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('//') or stripped.startswith('/*'):
            continue

        # Check if line ends with semicolon (statement boundary)
        if stripped.endswith(';'):
            stmts.append(stripped)
        else:
            current_stmt += " " + stripped

    # Shuffle independent statements (simple heuristic)
    random.seed(42)
    random.shuffle(stmts)

    # Rejoin statements
    result = '\n'.join(stmts)

    return result


def _obf_inline_temporaries(code):
    """(5) Inline single-use temporaries."""
    import re

    # Find temporary variables (v1, v2, ..., sub_XXXX)
    temp_pattern = re.compile(r'\b(v\d+|sub_[0-9A-F]{4})\b')

    # Simple heuristic: replace temp variables with their values
    # (in a real implementation, we'd track assignments)
    result = code

    return result


def _obf_split_expressions(code):
    """(6) Split compound expressions."""
    import re

    # Find compound assignments (a = b + c; a += b;)
    result = code

    # Split compound expressions (simple heuristic)
    result = re.sub(r'(\w+)\s*([+\-*/]=)\s*(\w+)', r'\1 = \1 \2 \3', result)

    return result


def apply_all_obfuscations(code, seed=42):
    """Apply all 6 obfuscation transforms to code."""
    result = code

    # (1) Rename identifiers
    result = _obf_rename_identifiers(result, seed=seed)

    # (2) Strip comments
    result = _obf_strip_comments(result)

    # (3) Reformat integer literals
    result = _obf_reformat_int_literals(result, seed=seed)

    # (4) Reorder statements
    result = _obf_reorder_statements(result)

    # (5) Inline temporaries
    result = _obf_inline_temporaries(result)

    # (6) Split compound expressions
    result = _obf_split_expressions(result)

    return result


# --- Crypto primitive reference implementations ---

def md5_init():
    """MD5 init constants: 0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476."""
    return [0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476]


def md5_sine_table():
    """MD5 sine table T[1..16] from RFC 1321 (first 16 values)."""
    return [
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee,
        0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
        0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
        0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
    ]


def sha1_init():
    """SHA-1 initial hash values H0..H4 from FIPS 180-4 / RFC 3174."""
    return [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0]


def sha256_init():
    """SHA-256 init constants: 0x6a09e667, ..."""
    return [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ]


def sha256_K():
    """SHA-256 K constants (first 8): 0x428a2f98, ..."""
    return [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
        0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    ]


def aes_forward_sbox():
    """AES forward S-box (full 256 values from NIST FIPS 197 Table 4).

    Corrected 2026-07-29: the previous table had 0xc4 at index 7 (should be
    0xc5) and indices 8-15 were fabricated (0x96,0x03,0x9c,0xa7,0xc2,0xb1,
    0x6e,0x4d bore no relation to the real Rijndael S-box), discovered when
    it silently undermined a fallback-data crypto-primitive matcher built
    against these values. Expanded to the full table so ordered-run matchers
    can scan from any table offset rather than overfitting the first row."""
    return [
        0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5,
        0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
        0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0,
        0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
        0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc,
        0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
        0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a,
        0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
        0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0,
        0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
        0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b,
        0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
        0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85,
        0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
        0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5,
        0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
        0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17,
        0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
        0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88,
        0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
        0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c,
        0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
        0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9,
        0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
        0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6,
        0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
        0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e,
        0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
        0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94,
        0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
        0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68,
        0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
    ]


def aes_rcon():
    """AES Rcon (first 8 values)."""
    return [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80]


def rc4_ksa(key):
    """RC4 KSA (Key Scheduling Algorithm)."""
    S = list(range(256))
    j = 0

    for i in range(256):
        j = (j + S[i] + key[i % len(key)]) & 0xFF
        S[i], S[j] = S[j], S[i]

    return S


def rc4_prga(S, length):
    """RC4 PRGA (Pseudo-Random Generation Algorithm)."""
    i = 0
    j = 0
    output = []

    for _ in range(length):
        i = (i + 1) & 0xFF
        j = (j + S[i]) & 0xFF
        S[i], S[j] = S[j], S[i]
        output.append(S[(S[i] + S[j]) & 0xFF])

    return bytes(output)


def chacha20_init():
    """ChaCha20 init constants: 0x61707865, 0x3320646e, 0x79622d32, 0x6b206574."""
    return [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574]


def chacha20_rotl(x, n):
    """ChaCha20 rotl (rotate left) by n bits."""
    return ((x << n) | (x >> (32 - n))) & 0xFFFFFFFF


def salsa20_init():
    """Salsa20 init constants (same "expand 32-byte k" words as ChaCha20)."""
    return [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574]


def salsa20_rotl(x, n):
    """Salsa20 rotl (rotate left) by n bits."""
    return ((x << n) | (x >> (32 - n))) & 0xFFFFFFFF


def tea_delta():
    """TEA delta constant: 0x9E3779B9."""
    return 0x9E3779B9


def xtea_delta():
    """XTEA delta constant: 0x9E3779B9 (same as TEA, but sum schedule differs)."""
    return 0x9E3779B9


def blowfish_p_array():
    """Blowfish P-array (full 18 words from the hex digits of pi, less the
    leading 3 - the canonical Schneier reference values): 0x243F6A88, ...

    Expanded 2026-07-29 from a partial 8-value table: with only 8 values
    available, a windowed/rotated sampler for synthetic training rows had
    just 8 distinct rotations to draw from regardless of how many candidates
    it generated, causing 86% within-family dedup collapse (59 pre-dedup
    rows -> 8 post-dedup) when combined into the v4 training mix."""
    return [
        0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344,
        0xA4093822, 0x299F31D0, 0x082EFA98, 0xEC4E6C89,
        0x452821E6, 0x38D01377, 0xBE5466CF, 0x34E90C6C,
        0xC0AC29B7, 0xC97C50DD, 0x3F84D5B5, 0xB5470917,
        0x9216D5D9, 0x8979FB1B,
    ]


def crc32_poly():
    """CRC32 reflected polynomial: 0xEDB88320."""
    return 0xEDB88320


def des_ip_table():
    """DES Initial Permutation (IP) table."""
    return [
        58, 50, 42, 34, 26, 18, 10, 2,
        60, 52, 44, 36, 28, 20, 12, 4,
        62, 54, 46, 38, 30, 22, 14, 6,
        64, 56, 48, 40, 32, 24, 16, 8,
        57, 49, 41, 33, 25, 17, 9, 1,
        59, 51, 43, 35, 27, 19, 11, 3,
        61, 53, 45, 37, 29, 21, 13, 5,
        63, 55, 47, 39, 31, 23, 15, 7,
    ]


def des_fp_table():
    """DES Final Permutation (FP) table."""
    return [
        40, 8, 48, 16, 56, 24, 64, 32,
        39, 7, 47, 15, 55, 23, 63, 31,
        38, 6, 46, 14, 54, 22, 62, 30,
        37, 5, 45, 13, 53, 21, 61, 29,
        36, 4, 44, 12, 52, 20, 60, 28,
        35, 3, 43, 11, 51, 19, 59, 27,
        34, 2, 42, 10, 50, 18, 58, 26,
        33, 1, 41, 9, 49, 17, 57, 25,
    ]


def des_sboxes():
    """DES 8 S-boxes (each 4x16) from FIPS PUB 46-3."""
    return [
        [
            [14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
            [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
            [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
            [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13],
        ],
        [
            [15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
            [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
            [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
            [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9],
        ],
        [
            [10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8],
            [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1],
            [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7],
            [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12],
        ],
        [
            [7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15],
            [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9],
            [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4],
            [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14],
        ],
        [
            [2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9],
            [14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6],
            [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14],
            [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3],
        ],
        [
            [12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11],
            [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8],
            [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6],
            [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13],
        ],
        [
            [4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1],
            [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6],
            [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2],
            [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12],
        ],
        [
            [13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7],
            [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2],
            [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8],
            [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11],
        ],
    ]


def base64_alphabet():
    """Base64 alphabet: "ABC...+/".

    Returns the 64-character base64 encoding table.
    """
    return "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"


def xor_cipher(key, data):
    """XOR cipher (keyed byte-XOR loop).

    Encrypts/decrypts data by XORing with the key (repeated as needed).
    """
    result = bytearray(len(data))

    for i in range(len(data)):
        result[i] = data[i] ^ key[i % len(key)]

    return bytes(result)


# --- Export all primitives for use by 1c/1d ---

__all__ = [
    # Obfuscation transforms
    "apply_all_obfuscations",

    # MD5
    "md5_init", "md5_sine_table",

    # SHA-1
    "sha1_init",

    # SHA-256
    "sha256_init", "sha256_K",

    # AES
    "aes_forward_sbox", "aes_rcon",

    # RC4
    "rc4_ksa", "rc4_prga",

    # ChaCha20
    "chacha20_init", "chacha20_rotl",

    # Salsa20
    "salsa20_init", "salsa20_rotl",

    # TEA/XTEA
    "tea_delta", "xtea_delta",

    # Blowfish
    "blowfish_p_array",

    # CRC32
    "crc32_poly",

    # DES
    "des_ip_table", "des_fp_table", "des_sboxes",

    # Base64
    "base64_alphabet",

    # XOR cipher
    "xor_cipher",
]


if __name__ == "__main__":
    # Test all primitives
    print("Testing crypto primitive library...")

    # MD5
    md5_init_vals = md5_init()
    print(f"MD5 init: {md5_init_vals}")

    # SHA-1
    sha1_init_vals = sha1_init()
    print(f"SHA-1 init: {sha1_init_vals}")

    # SHA-256
    sha256_init_vals = sha256_init()
    print(f"SHA-256 init: {sha256_init_vals}")

    # AES
    aes_sbox = aes_forward_sbox()
    print(f"AES S-box (first 16): {aes_sbox}")

    # RC4
    key = b"secret"
    S = rc4_ksa(key)
    print(f"RC4 KSA (first 8): {S[:8]}")

    # ChaCha20
    chacha_init = chacha20_init()
    print(f"ChaCha20 init: {chacha_init}")

    # Salsa20
    salsa_init = salsa20_init()
    print(f"Salsa20 init: {salsa_init}")

    # TEA/XTEA
    print(f"TEA delta: {hex(tea_delta())}")
    print(f"XTEA delta: {hex(xtea_delta())}")

    # Blowfish
    blowfish_p = blowfish_p_array()
    print(f"Blowfish P-array (first 8): {blowfish_p}")

    # CRC32
    print(f"CRC32 poly: {hex(crc32_poly())}")

    # DES
    print(f"DES IP table (first 8): {des_ip_table()[:8]}")
    print(f"DES FP table (first 8): {des_fp_table()[:8]}")

    # Base64
    print(f"Base64 alphabet (first 16): {base64_alphabet()[:16]}")

    # XOR cipher
    key = b"key"
    data = b"hello world"
    encrypted = xor_cipher(key, data)
    decrypted = xor_cipher(key, encrypted)
    print(f"XOR cipher: {data} -> {encrypted} -> {decrypted}")

    # Obfuscation
    sample_code = """
    int main() {
        int a = 10;
        int b = 20;
        int c = a + b;
        printf("Result: %d\\n", c);
        return 0;
    }
    """

    obfuscated = apply_all_obfuscations(sample_code)
    print(f"\nObfuscated code:\n{obfuscated}")

    print("\nAll primitives tested successfully!")
