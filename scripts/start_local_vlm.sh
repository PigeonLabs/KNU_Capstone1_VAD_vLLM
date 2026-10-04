#!/usr/bin/env bash
set -euo pipefail
# parallel=1 is the user's explicit exception to remaining llama.cpp defaults.
exec "${LLAMA_SERVER_BIN:-/home/jeong/Server/llama.cpp/build/bin/llama-server}" \
  -m "${QWEN_MODEL:-/home/jeong/Models/Qwen3.8-27B/Qwen3.8-27B-Q8_0.gguf}" \
  --mmproj "${QWEN_MMPROJ:-/home/jeong/Models/Qwen3.8-27B/mmproj-Qwen3.8-27B-Q8_0.gguf}" \
  --ctx-size 65536 --temp 0.2 --gpu-layers all --reasoning on --parallel 1
