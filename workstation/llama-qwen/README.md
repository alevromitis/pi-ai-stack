# Qwen on the workstation (the default model)

New chats default to **Qwen3.8-27B** (Unsloth UD-IQ2_XXS, since 2026-11-13;
Qwen3.5-9B Q4_K_M before) on the workstation's Arc B570,
served by llama.cpp's `llama-server` (Vulkan) as the systemd *user* unit
[`llama-qwen.service`](llama-qwen.service) on port 8091 of the
workstation's tailnet address (tailnet only, API key required). The unit file
explains each flag. In short: all layers on the GPU (~8.6 GiB of VRAM, under
a 9600 MB budget), 20k context, f16 KV cache, flash attention off (2.5x
slower on this Vulkan build), `-ub 256`, the vision projector on the CPU,
one slot, Qwen's recommended thinking-mode sampling, reasoning effort medium
(the template defaults to xhigh, which thinks too long for 20k). That gives ~15 tok/s
generation and ~430 tok/s prefill (14 / 190 with 16k in context;
2026-11-13). The 9B did ~42 and ~1000.

```bash
install -Dm644 workstation/llama-qwen/llama-qwen.service ~/.config/systemd/user/
( umask 077; mkdir -p ~/.config/llama-qwen; openssl rand -hex 32 > ~/.config/llama-qwen/api-keys )
echo "LLAMA_QWEN_HOST=$(tailscale ip -4)" > ~/.config/llama-qwen/env
systemctl --user daemon-reload && systemctl --user enable --now llama-qwen
```

`~/.config/llama-qwen/env` holds the address to bind (`LLAMA_QWEN_HOST`),
so the unit file carries none. Without it the unit fails to start.

LibreChat's `Qwen` custom endpoint calls it directly over the tailnet
(`QWEN_BASE_URL`, `QWEN_API_KEY` in `.env.librechat`; the key is also a line
of `~/.config/llama-qwen/api-keys` on the workstation). There is no
fallback: while the workstation is off, Qwen chats fail, and you switch to
the LiteRT-LM spec or an E2B preset by hand. The unit runs only while the
user is logged in; `loginctl enable-linger` would keep it up without a login.

* **History.** Disabled 2026-10-31 at the user's request (the kernel's OOM
  killer had also taken it on 2026-10-30 and 10-31: ~5.2 GB host RAM peak,
  `OOMScoreAdjust=500`); enabled again 2026-11-06.
* **Defaults.** The `qwen-chat` spec is the soft default: first chats, and
  chats after clearing. It attaches the Qwen-sized Wikipedia and web search
  (`wikipedia-large`, `searxng-large`: longer passages, whole sections,
  paged web pages), the calculator and the Debian sandbox; Playwright and
  translate are added from the input bar. `qwen-organiser` is Qwen with the
  calendar, tasks, notes and feeds, kept apart from the web tools (see the
  comment in `librechat.yaml`). The MongoDB preset
  **Qwen 27B - Thinker** is a copy of it and is the user's default preset.
  LibreChat's own default-preset path ignores the spec, and presets carry no
  MCP selection. The memory agent stays on LiteRT-LM, whose memory gate
  answers most of its requests without the model.
* **Tools.** Every MCP server works with Qwen. llama-server parses Qwen's XML
  tool calls into OpenAI `tool_calls`, several per step. The Pi-only crutches
  (`citations.py`, `reminder.py`) are not needed and apply only while the Pi
  answers.
* **Context.** 20480 Qwen tokens server-side (the user's choice; 32k also
  fits, 64k only with flash attention and a q8 KV at 5.8 tok/s).
  `tokenConfig` and the specs set 19456. `qwen-chat` gives the reply
  `max_tokens: 8192`, thinking included, so the prompt, history and tool
  results get ~11k (`qwen-organiser`: 6144 and ~13k). Hard problems may
  think past 8192 and be cut off. On 2026-11-13 the 22 existing Qwen chats
  were set to the same limits (backup: `~/librechat-qwen-backup-20261113.json`
  on rpi). Before: 49152 server-side, 47104 / 20480 for the 9B.
