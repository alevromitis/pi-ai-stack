"""Fixes LiteRT-LM 0.18.0's speculative decoding (MTP) rollback in the wheel's
own liblitert-lm.so (google-ai-edge/LiteRT-LM#3439), one instruction.

When a speculative step returns several tokens and the stop token is not the
last of them, runtime/core/tasks.cc rolls the executor back by
`sequence_length - step`. The executor's step counts its pending input token,
which must be the stop token, so the right amount is one less: as it is, every
such turn loses its stop token from the KV cache, and the prefix cache
continues those conversations. Measured on the Pi (2026-10-12, 6 greedy turns):
the conversation's token count after each turn was 21 39 68 88 with MTP off,
20 37 65 84 with MTP on, and 21 39 68 88 with MTP on and this fix.
litert-lm-v0.18.0-mtp-rollback.patch is the same fix in source.

Why the binary and not a source build: the open-source build can't use
Google's prebuilt tool-call constraint provider (built against their libc++;
it traps on the first tool call), and the PyPI library has it built in.

In the wheel's Tasks::Decode (DecodeOneStep::Run inlined), the rollback is

  0x440eaa0  cmp  x0, #0x1             ; GetCurrentStep() status is OK
  0x440eaa4  b.ne <error>              ; (taken only on error)
  ...                                  ; no flag-setting instructions
  0x440eab4  sub  w8, w26, w8          ; step - sequence_length
  0x440eabc  add  w1, w9, w8           ; current_step - (sequence_length - step)
  0x440eac4  blr  x10                  ; SetCurrentStep(w1)

The carry flag is set there (x0 == 1 after `cmp x0, #1`), so `adc w1, w9, w8`
gives current_step - (sequence_length - step - 1). It is the binary's only
call to StopTokenDetector::AllDone, and nothing branches into that window.

usage: fix_mtp_rollback.py <path to liblitert-lm.so>
Fails (non-zero) unless the file is exactly the 0.18.0 aarch64 wheel's
library, before and after.
"""

import hashlib
import sys

_STOCK_SHA256 = "b0c0258edc9e4868bd62f18526bb32791739a1d1b93df9360c8fa726814571ff"
_OFFSET = 0x440EABC
_ADD_W1_W9_W8 = bytes.fromhex("2101080b")
_ADC_W1_W9_W8 = bytes.fromhex("2101081a")

path = sys.argv[1]
data = bytearray(open(path, "rb").read())
digest = hashlib.sha256(data).hexdigest()
if digest != _STOCK_SHA256:
  sys.exit(f"fix_mtp_rollback: {path} is not the 0.18.0 aarch64 library ({digest})")
if data[_OFFSET:_OFFSET + 4] != _ADD_W1_W9_W8:
  sys.exit(f"fix_mtp_rollback: unexpected bytes at {_OFFSET:#x}")
data[_OFFSET:_OFFSET + 4] = _ADC_W1_W9_W8
with open(path, "wb") as f:
  f.write(data)
print(f"fix_mtp_rollback: patched {path} -> {hashlib.sha256(data).hexdigest()}")
