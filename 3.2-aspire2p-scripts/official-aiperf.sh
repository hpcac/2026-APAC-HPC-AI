#!/usr/bin/env bash
# The official AIPerf command from "1. Qwen3 AI Task Rules.md", with the
# concurrency, router address and served model name passed in:
#   N=<concurrency> HOST=<router host:port> MODEL=<served model name> bash official-aiperf.sh
# Do not change the aiperf parameters for the official scored re-run.
set -euo pipefail

: "${N:?team-declared concurrency}"
: "${HOST:?gateway host:port}"
: "${MODEL:?team /v1/models id}"

exec aiperf profile \
  --url "http://${HOST}" \
  --endpoint-type chat \
  --endpoint /v1/chat/completions \
  --model "${MODEL}" \
  --tokenizer Qwen/Qwen3-235B-A22B-Instruct-2507 \
  --streaming \
  -H 'Accept: text/event-stream' \
  --concurrency "${N}" \
  --workers-max "${N}" \
  --api-key NOT_USED \
  --warmup-duration 60 \
  --benchmark-duration 120 \
  --benchmark-grace-period 30 \
  --synthetic-input-tokens-mean 4000 \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 200 \
  --output-tokens-stddev 0 \
  --extra-inputs min_tokens:200 \
  --extra-inputs max_tokens:200 \
  --extra-inputs ignore_eos:true \
  --extra-inputs temperature:0.0 \
  --extra-inputs repetition_penalty:1.0 \
  --num-dataset-entries 12800 \
  --random-seed 2026 \
  --cache-bust first_turn_prefix \
  --goodput "time_to_first_token:3000 inter_token_latency:20" \
  --ui simple
