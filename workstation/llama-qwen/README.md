# Qwen on the workstation (the default model)

New chats default to **Qwen3.5-9B** (Q4_K_M) on the workstation's Arc B570,
served by llama.cpp's `llama-server` (Vulkan) as the systemd *user* unit
[`llama-qwen.service`](llama-qwen.service) on port 8091 of the
workstation's tailnet address (tailnet only, API key required). The unit file
explains each flag. In short: 48k context, f16 KV cache, flash attention off
(slower on this Vulkan build), one slot, Qwen's recommended thinking-mode
sampling. That gives ~42 tok/s generation and ~1000 tok/s prefill (2026-10-10).

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
  **Qwen 9B - Thinker** is a copy of it and is the user's default preset.
  LibreChat's own default-preset path ignores the spec, and presets carry no
  MCP selection. The memory agent stays on LiteRT-LM, whose memory gate
  answers most of its requests without the model.
* **Tools.** Every MCP server works with Qwen. llama-server parses Qwen's XML
  tool calls into OpenAI `tool_calls`, several per step. The Pi-only crutches
  (`citations.py`, `reminder.py`) are not needed and apply only while the Pi
  answers.
* **Context.** 49152 Qwen tokens server-side; 64k left the desktop too
  little VRAM (decode fell to 20 tok/s). `tokenConfig` and the spec set
  47104. `qwen-chat` gives the reply `max_tokens: 20480`, thinking included,
  so history and tool results get up to 26624 (`qwen-organiser`: 16384 and
  30720). It was Qwen's recommended 32768 until 2026-11-06, which left
  14336: too little for the larger tools (~3-4k tokens a call). Hard maths
  took 8-14k tokens on this card (2026-10-10).
