#!/usr/bin/env python3
"""Phase 1d - Build ransomware-crypto: vuln + key_recovery.

Self-contained SYNTHETIC generator (no external HF dataset). Each row presents a
decompiled ransomware crypto routine with an INJECTED, real-world weakness, and
the assistant names the weakness and gives a concrete key-recovery / decryptor
strategy. Incident-response content: identify the flaw, recover the key.

Output: datasets.DatasetDict with train (~3,500 rows), family="ransomware-crypto".
"""

import os
import random
from datasets import Dataset, DatasetDict

SEED = 42
TARGET_ROWS = 3_500

SYSTEM = (
    "You are a ransomware incident-response cryptanalyst. Given a decompiled "
    "encryption routine, identify the exploitable weakness and give a concrete, "
    "correct key-recovery or decryption strategy. Cite the specific evidence."
)

# (weakness_name, decompiled_snippet [uses @KEY@/@INC@ tokens, NOT str.format],
#  recovery_strategy)
VULNS = [
    ("hardcoded symmetric key",
     "void encrypt(uint8_t*buf,size_t n){ static const uint8_t k[16]={@KEY@};\n"
     "  for(size_t i=0;i<n;i++) buf[i]^=k[i%16]; }",
     "The 16-byte key is embedded in the binary as a static array. Extract those "
     "bytes directly from .rodata and XOR the ciphertext to recover plaintext for "
     "every victim - no brute force needed."),
    ("ECB mode (no diffusion)",
     "AES_set_encrypt_key(k,128,&ks);\n"
     "for(i=0;i<n;i+=16) AES_ecb_encrypt(pt+i,ct+i,&ks,1);",
     "ECB encrypts identical 16-byte blocks to identical ciphertext. Known-plaintext "
     "on the file header/magic recovers block mappings; structure leaks without the key, "
     "and a header crib often yields the key schedule when combined with the static IV bug."),
    ("time-seeded PRNG key",
     "srand((unsigned)time(0));\nfor(i=0;i<32;i++) key[i]=rand()&0xff;",
     "The key derives from srand(time(0)). The file mtime bounds the seed to a small "
     "window; brute-force the ~10^3 candidate seconds, regenerate each rand() stream, "
     "and trial-decrypt against the known header. Recovers the key in seconds."),
    ("static / reused IV",
     "static uint8_t iv[16]={0};\nAES_cbc_encrypt(pt,ct,n,&ks,iv,1);",
     "A fixed all-zero IV reused across files means identical plaintext prefixes yield "
     "identical ciphertext prefixes. Combined with a known header this leaks the first "
     "block's keystream; cross-file correlation enables partial recovery without the key."),
    ("keystream reuse (two-time pad)",
     "gen_keystream(ks,n);\nfor(i=0;i<n;i++) ct[i]=pt[i]^ks[i]; /* ks reused per victim */",
     "The same keystream encrypts multiple files (two-time pad). XOR two ciphertexts to "
     "cancel the keystream: ct1^ct2 = pt1^pt2. Crib-drag known file structures to peel "
     "apart both plaintexts, then recover the keystream itself."),
    ("weak XOR rolling key",
     "uint8_t k=seed;\nfor(i=0;i<n;i++){ buf[i]^=k; k=(k+@INC@)&0xff; }",
     "A single-byte rolling XOR with a linear update has 256x256 possible (seed,inc) "
     "pairs. Brute-force all 65536 combinations against the known header magic; the "
     "correct pair decrypts the whole file. Trivially recoverable."),
]


def build_ransomware_crypto_dataset(cap=0):
    """Build the ransomware-crypto dataset (synthetic, self-contained)."""
    random.seed(SEED)
    n = TARGET_ROWS if cap <= 0 else min(cap, TARGET_ROWS)
    rows = []
    for i in range(n):
        name, snippet, recovery = VULNS[i % len(VULNS)]
        keybytes = ",".join(f"0x{random.randint(0,255):02x}" for _ in range(16))
        code = snippet.replace("@KEY@", keybytes).replace("@INC@", str(random.choice([1, 3, 7, 13, 17])))
        user = (
            "This encryption routine was recovered from a ransomware sample. Identify the "
            f"exploitable weakness and give a key-recovery strategy.\n\n```c\n{code}\n```"
        )
        assistant = f"**Weakness: {name}.** {recovery}"
        rows.append({
            "source": "synthetic/ransomware-crypto",
            "family": "ransomware-crypto",
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
        })
    ds_out = DatasetDict({"train": Dataset.from_list(rows)})
    print(f"Built ransomware-crypto dataset: {len(rows)} rows")
    return ds_out


if __name__ == "__main__":
    cap = int(os.environ.get("CAP", "0"))
    ds = build_ransomware_crypto_dataset(cap=cap)
    print(f"Train: {len(ds['train'])} rows")
    print("ransomware-crypto dataset ready (family=\"ransomware-crypto\")")
