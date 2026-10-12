"""Gives LiteRT-LM 0.18.0's C API a way to prefill a conversation's preface
(system prompt, tools, history) when the conversation is created, one
instruction in the wheel's own liblitert-lm.so.

C++ has ConversationConfig::Builder::SetPrefillPrefaceOnInit, but the 0.18.0
C API has no setter for it: a conversation's system prompt is prefilled
together with its first message, so patches/prefix_cache.py has no "system
prompt only" state to clone for new chats (PLAN-prefix-clone.md).

The C API's filter_channel_content_from_kv_cache setter is one serve never
calls. litert_lm_conversation_create (c/conversation.cc) copies it into the
Builder, which sits at sp+1304:

  0x3379f0c  ldrb w9, [x20, #378]   ; c_config's optional<bool> is set
  0x3379f14  cbz  w9, <skip>
  0x3379f18  ldrb w9, [x20, #377]   ; its value
  0x3379f1c  strb w9, [sp, #2248]   ; builder +944, filter_channel_content_from_kv_cache_

Storing to [sp, #2153] instead, builder +849, which is
prefill_preface_on_init_ (Builder::Build reads it at [x19, #849], next to
enable_constrained_decoding_ at +848), turns that setter into "prefill the
preface on init". In Python,
create_conversation(filter_channel_content_from_kv_cache=True) prefills the
preface at creation, and the filter keeps its default (on). Conversations
created without the argument are unchanged.

usage: prefill_preface_option.py <path to liblitert-lm.so>
Run after fix_mtp_rollback.py. Fails (non-zero) unless the file is exactly
the library that leaves, before and after.
"""

import hashlib
import sys

_MTP_FIXED_SHA256 = "e94492655e4d0f84070a7edc11bff865ff9f8df498e3813e2e3cad8ff0b715fa"
_OFFSET = 0x3379F1C


def _strb_sp(rt: int, imm12: int) -> bytes:
  """STRB (immediate, unsigned offset) wRt, [sp, #imm12]."""
  return (0x39000000 | imm12 << 10 | 31 << 5 | rt).to_bytes(4, "little")


_STORE_FILTER = _strb_sp(9, 2248)  # 392323e9
_STORE_PREFILL = _strb_sp(9, 2153)  # 3921a7e9

path = sys.argv[1]
data = bytearray(open(path, "rb").read())
digest = hashlib.sha256(data).hexdigest()
if digest != _MTP_FIXED_SHA256:
  sys.exit(f"prefill_preface_option: {path} is not the 0.18.0 aarch64 library"
           f" with fix_mtp_rollback.py applied ({digest})")
if data[_OFFSET:_OFFSET + 4] != _STORE_FILTER:
  sys.exit(f"prefill_preface_option: unexpected bytes at {_OFFSET:#x}")
data[_OFFSET:_OFFSET + 4] = _STORE_PREFILL
with open(path, "wb") as f:
  f.write(data)
print(f"prefill_preface_option: patched {path} -> {hashlib.sha256(data).hexdigest()}")
