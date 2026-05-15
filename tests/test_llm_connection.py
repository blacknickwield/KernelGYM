#!/usr/bin/env python3
"""Smoke test for AsyncLLMClient in synthesize_optimizations_openai.py.

Usage:
  export OPENAI_API_KEY=sk-...
  python test_llm_connection.py \
    --script ./synthesize_optimizations_openai.py \
    --llm-model gpt-4.1 \
    --llm-api https://api.openai.com/v1

This script only tests whether the OpenAI-compatible LLM endpoint can be reached
and whether AsyncLLMClient.call(...) returns non-empty text.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any


def load_pipeline_module(script_path: str) -> Any:
    """Load synthesize_optimizations_openai.py from an arbitrary path."""
    path = Path(script_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Script not found: {path}")

    spec = importlib.util.spec_from_file_location("synthesize_optimizations_openai", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to import module from: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


async def main() -> int:
    parser = argparse.ArgumentParser(description="Test OpenAI LLM connectivity for AsyncLLMClient")
    parser.add_argument(
        "--script",
        default="synthesize_optimizations_openai.py",
        help="Path to the pipeline script containing PipelineConfig and AsyncLLMClient",
    )
    parser.add_argument(
        "--llm-api",
        default="https://api.openai.com/v1",
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-4.1",
        help="Model name to test",
    )
    parser.add_argument(
        "--llm-key",
        default=None,
        help="API key. If omitted, uses OPENAI_API_KEY from environment.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--rpm", type=int, default=30)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--prompt",
        default="Reply with exactly one word: pong",
        help="Prompt used for the smoke test",
    )
    args = parser.parse_args()

    api_key = args.llm_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[FAIL] Missing API key. Set OPENAI_API_KEY or pass --llm-key.")
        return 2

    module = load_pipeline_module(args.script)

    config = module.PipelineConfig(
        llm_api_url=args.llm_api,
        llm_model=args.llm_model,
        llm_api_key=api_key,
        llm_temperature=args.temperature,
        llm_max_tokens=args.max_tokens,
        llm_rpm=args.rpm,
        llm_concurrency=args.concurrency,
    )

    client = module.AsyncLLMClient(max_concurrent=args.concurrency)
    try:
        print(f"[INFO] API base: {config.llm_api_url}")
        print(f"[INFO] Model:    {config.llm_model}")
        print("[INFO] Sending test request...")

        text = await client.call(args.prompt, config)

        if text is None:
            print("[FAIL] LLM call returned None. Check API key, base URL, model name, or network.")
            return 1

        if not text.strip():
            print("[FAIL] LLM call returned an empty response.")
            return 1

        print("[OK] LLM communication works.")
        print(f"[OK] Response: {text.strip()!r}")
        return 0
    finally:
        await client.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
