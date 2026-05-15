#!/usr/bin/env python3
"""
CUDA/Triton Kernel Optimization Data Synthesis Pipeline.

Uses CUDA-Agent-Ops-6K (BytedTsinghua-SIA) as raw reference data
and KernelGYM as the evaluation engine to produce a verified
dataset of (reference_code, optimized_kernel, speedup, profiling_meta) pairs.

Usage:
  # 1. Start KernelGYM server + workers first:
  #    bash start_all_with_monitor.sh
  #
  # 2. Run synthesis:
  #    python scripts/synthesize_optimizations.py \
  #        --strategy llm \
  #        --llm-api https://api.openai.com/v1 \
  #        --llm-model gpt-4.1 \
  #        --llm-key $OPENAI_API_KEY \
  #        --max-samples 100 \
  #        --output ./synthesized_data
  #
  # 3. Refine mode (iterative feedback):
  #    python scripts/synthesize_optimizations.py \
  #        --strategy refine-llm \
  #        --refine-rounds 5 \
  #        --max-samples 50
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import json
import os
import sys
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx
import openai

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    # KernelGYM server
    kgym_url: str = "http://localhost:10907"

    # Dataset
    dataset_name: str = "BytedTsinghua-SIA/CUDA-Agent-Ops-6K"
    dataset_path: Optional[str] = None  # local parquet/directory path, skips HF download
    max_samples: int = 0  # 0 = all
    data_source_filter: Optional[str] = None  # e.g. "torch#3", None = all

    # Generation strategy
    strategy: str = "llm"  # "llm", "refine-llm", "torch_compile", "passthrough", "custom"
    llm_api_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4.1"
    llm_api_key: Optional[str] = None
    llm_temperature: float = 0.6
    llm_max_tokens: int = 16384
    num_candidates_per_task: int = 1
    refine_rounds: int = 3
    refine_concurrency: int = 5  # max parallel tasks in refine mode

    # KernelGYM evaluation params
    backend: str = "triton"
    num_correct_trials: int = 3
    num_perf_trials: int = 50
    timeout: int = 120
    enable_profiling: bool = True
    batch_size: int = 20

    # Filtering
    min_speedup: float = 0.95
    require_coverage: float = 0.0

    # Output
    output_dir: str = "./synthesized_data"
    checkpoint_file: Optional[str] = None
    resume: bool = True

    # Custom strategy
    custom_strategy_path: Optional[str] = None

    # Rate limiting
    llm_rpm: int = 30
    llm_concurrency: int = 3  # max concurrent LLM requests


# ---------------------------------------------------------------------------
# Async LLM Client (shared by all LLM strategies)
# ---------------------------------------------------------------------------

class AsyncLLMClient:
    """Async OpenAI SDK client with per-minute rate limiting and concurrency limiting.

    This replaces the previous hand-written httpx chat/completions client with
    the official `openai.AsyncOpenAI` client.  The public interface is kept
    compatible with the rest of this script: `call(prompt, config) -> Optional[str]`.
    """

    def __init__(self, max_concurrent: int = 3):
        self._request_count = 0
        self._window_start = time.monotonic()
        self._lock = asyncio.Lock()
        self._concurrency = asyncio.Semaphore(max_concurrent)
        self._client: Optional[openai.AsyncOpenAI] = None
        self._client_key: Optional[Tuple[Optional[str], Optional[str]]] = None

    async def _get_client(self, config: PipelineConfig) -> openai.AsyncOpenAI:
        """Create/reuse an AsyncOpenAI client for the current base_url/api_key."""
        api_key = config.llm_api_key or os.environ.get("OPENAI_API_KEY")
        base_url = config.llm_api_url.rstrip("/") if config.llm_api_url else None
        client_key = (base_url, api_key)

        if self._client is None or self._client_key != client_key:
            if self._client is not None:
                await self._client.close()

            kwargs: Dict[str, Any] = {"timeout": 300.0}
            if api_key:
                kwargs["api_key"] = api_key
            if base_url:
                kwargs["base_url"] = base_url

            self._client = openai.AsyncOpenAI(**kwargs)
            self._client_key = client_key

        return self._client

    async def _rate_limit(self, config: PipelineConfig) -> None:
        async with self._lock:
            self._request_count += 1
            if self._request_count >= config.llm_rpm:
                elapsed = time.monotonic() - self._window_start
                if elapsed < 60:
                    await asyncio.sleep(60 - elapsed + 0.5)
                self._request_count = 0
                self._window_start = time.monotonic()

    async def call(self, prompt: str, config: PipelineConfig) -> Optional[str]:
        """Send one OpenAI Chat Completions request and return response text."""
        await self._rate_limit(config)

        async with self._concurrency:
            client = await self._get_client(config)
            try:
                resp = await client.chat.completions.create(
                    model=config.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=config.llm_temperature,
                    max_tokens=config.llm_max_tokens,
                )

                choice = resp.choices[0] if resp.choices else None
                msg = choice.message if choice else None
                content = ((msg.content if msg and msg.content else "") or "").strip()

                # Some OpenAI-compatible endpoints may return non-standard
                # `reasoning_content`. Keep the old fallback behavior.
                reasoning = ""
                if msg is not None:
                    reasoning = (
                        getattr(msg, "reasoning_content", None)
                        or getattr(msg, "model_extra", {}).get("reasoning_content", "")
                        or ""
                    ).strip()

                if not content and reasoning:
                    usage = getattr(resp, "usage", None)
                    completion_tokens = getattr(usage, "completion_tokens", "unknown")
                    print(
                        f"  [LLM] content empty but reasoning_content has "
                        f"{len(reasoning)} chars. completion_tokens={completion_tokens}. "
                        f"Using reasoning_content as fallback."
                    )
                    return reasoning

                if not content:
                    usage = getattr(resp, "usage", None)
                    completion_tokens = getattr(usage, "completion_tokens", "unknown")
                    finish_reason = getattr(choice, "finish_reason", "unknown") if choice else "unknown"
                    print(
                        f"  [LLM] OpenAI API returned empty content. "
                        f"Prompt length: {len(prompt)} chars. "
                        f"completion_tokens={completion_tokens}. "
                        f"finish_reason={finish_reason}"
                    )

                return content

            except openai.APIError as e:
                print(f"  [LLM] OpenAI API error: {e}")
                return None
            except Exception as e:
                print(f"  [LLM] Unexpected OpenAI client error: {e}")
                return None

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
            self._client_key = None


# ---------------------------------------------------------------------------
# Generation Strategies
# ---------------------------------------------------------------------------


class GenerationStrategy(ABC):
    """Abstract base for kernel generation strategies (all methods are async)."""

    @abstractmethod
    async def generate(
        self, ref_code: str, ops: str, data_source: str, config: PipelineConfig
    ) -> List[str]:
        ...


class PassthroughStrategy(GenerationStrategy):
    """Returns reference code with Model renamed to ModelNew (for testing)."""

    async def generate(
        self, ref_code: str, ops: str, data_source: str, config: PipelineConfig
    ) -> List[str]:
        kernel_code = ref_code.replace("class Model(", "class ModelNew(")
        return [kernel_code] * config.num_candidates_per_task


class TorchCompileStrategy(GenerationStrategy):
    """Baselines: wraps the reference with torch.compile."""

    async def generate(
        self, ref_code: str, ops: str, data_source: str, config: PipelineConfig
    ) -> List[str]:
        kernel_code = ref_code.replace("class Model(", "class _RefModel(") + """

class ModelNew(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._ref = _RefModel(*args, **kwargs)
        self._compiled = torch.compile(self._ref)

    def forward(self, *args, **kwargs):
        return self._compiled(*args, **kwargs)
"""
        return [kernel_code] * config.num_candidates_per_task


class LLMGenerationStrategy(GenerationStrategy):
    """Uses an async LLM client to generate optimized Triton/CUDA kernels."""

    def __init__(self, llm_client: Optional[AsyncLLMClient] = None,
                 max_concurrent: int = 3):
        self._client = llm_client or AsyncLLMClient(max_concurrent=max_concurrent)

    TRITON_PROMPT_TEMPLATE = """You are a GPU kernel optimization expert. Your task is to optimize the given PyTorch code by writing a fused Triton kernel.

## Original PyTorch Operations
{ops}

## Reference PyTorch Code
```python
{ref_code}
```

## Instructions
1. Analyze what operations the reference code performs
2. Write an **optimized Triton kernel** that fuses as many operations as possible into a single GPU kernel
3. Your code MUST be a self-contained Python module with:
   - A class named `ModelNew` that inherits from `torch.nn.Module`
   - An `__init__` method that accepts the SAME arguments as the reference Model's __init__
   - A `forward` method that accepts and returns the SAME tensor shapes as the reference
4. Use `triton` and `triton.language as tl` for kernel definitions
5. The `__init__` constructor signature must match the reference's `Model.__init__` parameter list
6. Wrap your kernel in a proper Python class and call it from forward
7. Only output the Python code, no explanations

## Output Format
```python
import torch
import torch.nn as nn
import triton
import triton.language as tl

# Your optimized Triton kernel here

class ModelNew(nn.Module):
    def __init__(self, ...):
        super().__init__()
        ...

    def forward(self, ...):
        ...
```

Generate the optimized Triton kernel now:"""

    CUDA_PROMPT_TEMPLATE = """You are a GPU kernel optimization expert. Your task is to optimize the given PyTorch code by writing a fused CUDA kernel.

## Original PyTorch Operations
{ops}

## Reference PyTorch Code
```python
{ref_code}
```

## Instructions
1. Analyze what operations the reference code performs
2. Write an **optimized CUDA kernel** that fuses as many operations as possible
3. Your code MUST be a self-contained Python module with:
   - A class named `ModelNew` that inherits from `torch.nn.Module`
   - An `__init__` method that accepts the SAME arguments as the reference Model's __init__
   - A `forward` method that applies your optimized computation using CUDA
4. Use `torch.utils.cpp_extension.load_inline` or `torch.cuda` for CUDA kernel execution
5. The `__init__` constructor signature must match the reference's `Model.__init__` parameter list
6. Only output the Python code, no explanations

Generate the optimized CUDA kernel now:"""

    # ---- Refinement prompts (same structure, different focus) ----

    REFINE_COMPILATION_PROMPT = """Your previous Triton kernel failed to COMPILE. Fix the compilation error.

## Original Operations
{ops}

## Reference PyTorch Code
```python
{ref_code}
```

## Your Previous Kernel Code
```python
{prev_code}
```

## Compilation Error
```
{error}
```

## Instructions
Fix ALL compilation errors. Common issues: incorrect Triton syntax, type mismatches, missing imports, class name must be `ModelNew`, __init__ signature must match reference.
Output ONLY the corrected Python code. No explanations."""

    REFINE_RUNTIME_PROMPT = """Your previous Triton kernel compiled but produced a RUNTIME ERROR.

## Original Operations
{ops}

## Reference PyTorch Code
```python
{ref_code}
```

## Your Previous Kernel Code
```python
{prev_code}
```

## Runtime Error
```
{error}
```

## Instructions
Fix the runtime error. Common issues: shape mismatches, incorrect indexing, device mismatches, invalid memory access.
Output ONLY the corrected Python code with class ModelNew. No explanations."""

    REFINE_CORRECTNESS_PROMPT = """Your previous Triton kernel compiled and ran, but produced INCORRECT results.

## Original Operations
{ops}

## Reference PyTorch Code
```python
{ref_code}
```

## Your Previous Kernel Code
```python
{prev_code}
```

## Correctness Issue
The kernel output differs from the reference output. Possible causes: numerical precision, missing operations, wrong formula, edge cases.

## Instructions
Carefully review the reference code's operations and fix your Triton kernel to produce the CORRECT output.
Output ONLY the corrected Python code with class ModelNew. No explanations."""

    REFINE_PERFORMANCE_PROMPT = """Your previous Triton kernel compiled and produced correct results, but was SLOWER than the reference (speedup: {speedup:.2f}x).

## Original Operations
{ops}

## Reference PyTorch Code
```python
{ref_code}
```

## Your Previous Kernel Code
```python
{prev_code}
```

## Performance Data
- Reference runtime: {ref_runtime:.4f} ms
- Your kernel runtime: {kernel_runtime:.4f} ms
- Speedup: {speedup:.2f}x (target: > 1.0x)
{fusion_status}

## Instructions
Optimize for better performance: fuse more operations, use coalesced memory access, optimize grid/block sizes, reduce redundant computation, consider tiling.
Output ONLY the optimized Python code with class ModelNew. No explanations."""

    REFINE_DECOY_PROMPT = """Your previous Triton kernel compiled and ran, but did NOT actually use Triton kernels (decoy kernel detected).

## Original Operations
{ops}

## Reference PyTorch Code
```python
{ref_code}
```

## Your Previous Kernel Code (DECOY - no actual Triton usage)
```python
{prev_code}
```

## Instructions
Rewrite the kernel to ACTUALLY use Triton: define a triton.jit-decorated kernel, use triton.language (tl) operations, call the Triton kernel from ModelNew.forward().
Output ONLY the corrected Python code with class ModelNew. No explanations."""

    async def _call_llm(self, prompt: str, config: PipelineConfig) -> Optional[str]:
        return await self._client.call(prompt, config)

    @staticmethod
    def _extract_code(llm_output: str) -> str:
        if "```python" in llm_output:
            return llm_output.split("```python", 1)[1].split("```", 1)[0].strip()
        if "```" in llm_output:
            return llm_output.split("```", 1)[1].split("```", 1)[0].strip()
        return llm_output.strip()

    async def generate(
        self, ref_code: str, ops: str, data_source: str, config: PipelineConfig
    ) -> List[str]:
        template = (
            self.CUDA_PROMPT_TEMPLATE
            if config.backend == "cuda"
            else self.TRITON_PROMPT_TEMPLATE
        )
        prompt = template.format(ops=ops, ref_code=ref_code)

        async def _one_candidate(i: int) -> Optional[str]:
            raw = await self._call_llm(prompt, config)
            if raw is None:
                print(f"  [generate] LLM returned None for candidate {i}")
                return None
            code = self._extract_code(raw)
            if not code:
                print(f"  [generate] Extracted code is empty. "
                      f"Raw response length: {len(raw)} chars. "
                      f"Raw preview: {raw[:200]!r}")
                return None
            return code

        tasks = [_one_candidate(i) for i in range(config.num_candidates_per_task)]
        results = await asyncio.gather(*tasks)
        return [c for c in results if c]

    async def refine(
        self,
        ref_code: str,
        ops: str,
        prev_code: str,
        feedback: Dict[str, Any],
        config: PipelineConfig,
    ) -> Optional[str]:
        """Generate an improved kernel based on KGym evaluation feedback."""
        metadata = feedback.get("metadata", {}) or {}

        if not feedback.get("compiled", False):
            error = (
                metadata.get("compilation_error")
                or metadata.get("error")
                or metadata.get("validation_error")
                or "Unknown compilation error"
            )
            prompt = self.REFINE_COMPILATION_PROMPT.format(
                ops=ops, ref_code=ref_code, prev_code=prev_code,
                error=str(error)[:3000],
            )
            reason = "compilation_error"
        elif feedback.get("decoy_kernel", False):
            prompt = self.REFINE_DECOY_PROMPT.format(
                ops=ops, ref_code=ref_code, prev_code=prev_code,
            )
            reason = "decoy_kernel"
        elif not feedback.get("correctness", False):
            error = (
                metadata.get("runtime_error")
                or metadata.get("error")
                or metadata.get("correctness_issue")
                or "Output mismatch"
            )
            prompt = self.REFINE_RUNTIME_PROMPT.format(
                ops=ops, ref_code=ref_code, prev_code=prev_code,
                error=str(error)[:3000],
            )
            reason = "runtime_error"
        else:
            speedup = float(feedback.get("speedup", 0) or 0)
            ref_runtime = float(feedback.get("reference_runtime", 0) or 0)
            kernel_runtime = float(feedback.get("kernel_runtime", 0) or 0)
            cov = float(
                metadata.get("custom_kernel_cuda_time_coverage")
                or metadata.get("triton_kernel_coverage")
                or 0.0
            )
            fusion_status = (
                f"- Custom kernel coverage: {cov:.1%} — most time is NOT spent in your Triton kernels. Fuse more operations into Triton."
                if cov < 0.3
                else f"- Custom kernel coverage: {cov:.1%}"
            )
            prompt = self.REFINE_PERFORMANCE_PROMPT.format(
                ops=ops, ref_code=ref_code, prev_code=prev_code,
                speedup=speedup, ref_runtime=ref_runtime,
                kernel_runtime=kernel_runtime, fusion_status=fusion_status,
            )
            reason = f"slow_{speedup:.2f}x"

        print(f"    [Refine] reason: {reason}")
        raw = await self._call_llm(prompt, config)
        return self._extract_code(raw) if raw else None

    async def close(self):
        await self._client.close()


class CustomStrategy(GenerationStrategy):
    """Wraps a user-provided function (sync or async) loaded from a Python module path.

    Function signature:
        (ref_code: str, ops: str, data_source: str, config: PipelineConfig) -> List[str]
    Or async:
        async (ref_code: str, ops: str, data_source: str, config: PipelineConfig) -> List[str]
    """

    def __init__(self, func: Callable):
        self._func = func
        self._is_coro = asyncio.iscoroutinefunction(func)

    async def generate(
        self, ref_code: str, ops: str, data_source: str, config: PipelineConfig
    ) -> List[str]:
        if self._is_coro:
            return await self._func(ref_code, ops, data_source, config)
        return self._func(ref_code, ops, data_source, config)


def load_custom_strategy(path: str) -> GenerationStrategy:
    """Load a generation strategy from 'module.path:function_name'."""
    if ":" not in path:
        raise ValueError(f"Custom strategy path must be 'module:function', got: {path}")
    module_path, func_name = path.rsplit(":", 1)
    if module_path.endswith(".py"):
        spec = importlib.util.spec_from_file_location("_custom_strategy", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_path)
    return CustomStrategy(getattr(module, func_name))


# Strategy registry
STRATEGIES: Dict[str, GenerationStrategy] = {
    "passthrough": PassthroughStrategy(),
    "torch_compile": TorchCompileStrategy(),
    "llm": LLMGenerationStrategy(),
}


# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------


def load_dataset(config: PipelineConfig) -> List[Dict[str, Any]]:
    """Load and filter the CUDA-Agent-Ops-6K dataset.

    If config.dataset_path is set, loads from local parquet/directory.
    Otherwise downloads from HuggingFace.
    """
    from datasets import load_dataset as hf_load

    if config.dataset_path:
        path = config.dataset_path
        print(f"Loading dataset from local: {path}")
        if os.path.isdir(path):
            # Try recursive parquet glob first, fall back to non-recursive
            ds = hf_load("parquet", data_files=f"{path}/**/*.parquet", split="train")
        elif path.endswith(".parquet"):
            ds = hf_load("parquet", data_files=path, split="train")
        else:
            ds = hf_load(path, split="train")
    else:
        print(f"Loading dataset: {config.dataset_name}")
        ds = hf_load(config.dataset_name, split="train")

    print(f"  Total rows: {len(ds)}")

    rows = []
    for i, example in enumerate(ds):
        if config.max_samples > 0 and len(rows) >= config.max_samples:
            break
        source = example["data_source"]
        if config.data_source_filter and source != config.data_source_filter:
            continue
        rows.append({
            "idx": i,
            "ops": example["ops"],
            "data_source": source,
            "code": example["code"],
        })
    print(f"  Filtered to: {len(rows)} rows")
    return rows


# ---------------------------------------------------------------------------
# KernelGYM Client
# ---------------------------------------------------------------------------


def make_task_id(ref_idx: int, candidate_idx: int, strategy: str) -> str:
    short_hash = hashlib.md5(f"{ref_idx}:{candidate_idx}:{strategy}".encode()).hexdigest()[:8]
    return f"synth_{ref_idx}_{candidate_idx}_{short_hash}"


def build_evaluation_task(
    ref_code: str,
    kernel_code: str,
    task_id: str,
    config: PipelineConfig,
    entry_point: str = "Model",
) -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "reference_code": ref_code,
        "kernel_code": kernel_code,
        "backend": config.backend,
        "entry_point": entry_point,
        "num_correct_trials": config.num_correct_trials,
        "num_perf_trials": config.num_perf_trials,
        "timeout": config.timeout,
        "enable_profiling": config.enable_profiling,
        "enable_triton_detection": True,
    }


async def submit_batch(
    tasks: List[Dict[str, Any]],
    config: PipelineConfig,
) -> List[Dict[str, Any]]:
    """Submit a batch to KernelGYM /evaluate/batch (async)."""
    batch_id = f"synth_batch_{uuid.uuid4().hex[:8]}"
    payload = {"batch_id": batch_id, "tasks": tasks}
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{config.kgym_url}/evaluate/batch",
                json=payload,
                timeout=max(len(tasks) * config.timeout + 120, 600),
            )
            resp.raise_for_status()
            return resp.json().get("results", [])
    except Exception as e:
        print(f"  [KGym] batch submit error: {e}")
        return [
            {"task_id": t.get("task_id", "unknown"), "status": "failed", "error_message": str(e)}
            for t in tasks
        ]


async def submit_single(
    task: Dict[str, Any],
    config: PipelineConfig,
) -> Dict[str, Any]:
    """Submit a single task to KernelGYM /evaluate (async)."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{config.kgym_url}/evaluate",
                json=task,
                timeout=config.timeout + 60,
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        return {"task_id": task.get("task_id", "unknown"), "status": "failed", "error_message": str(e)}


# ---------------------------------------------------------------------------
# Quality Filtering
# ---------------------------------------------------------------------------


@dataclass
class QualifiedResult:
    task_id: str
    ref_code: str
    kernel_code: str
    ops: str
    data_source: str
    ref_idx: int
    compiled: bool
    correctness: bool
    decoy_kernel: bool
    reference_runtime: float
    kernel_runtime: float
    speedup: float
    metadata: Dict[str, Any] = field(default_factory=dict)


def filter_quality(
    results: List[Dict[str, Any]],
    ref_map: Dict[str, Tuple[int, str, str, str]],
    candidate_map: Dict[str, str],
    config: PipelineConfig,
) -> List[QualifiedResult]:
    qualified = []
    for r in results:
        task_id = r.get("task_id", "")
        ref_idx, ref_code, ops, data_source = ref_map.get(task_id, (None, "", "", ""))
        kernel_code = candidate_map.get(task_id, "")
        if r.get("status") not in ("completed", "failed"):
            continue

        compiled = bool(r.get("compiled", False))
        correctness = bool(r.get("correctness", False))
        decoy = bool(r.get("decoy_kernel", True))
        speedup = float(r.get("speedup", 0) or 0)
        ref_runtime = float(r.get("reference_runtime", 0) or 0)
        kernel_runtime = float(r.get("kernel_runtime", 0) or 0)
        meta = r.get("metadata", {}) or {}

        if not compiled or not correctness or decoy or speedup <= config.min_speedup:
            continue
        if config.require_coverage > 0:
            cov = meta.get("triton_kernel_coverage") or meta.get("custom_kernel_cuda_time_coverage") or 0
            if float(cov) < config.require_coverage:
                continue

        qualified.append(QualifiedResult(
            task_id=task_id, ref_code=ref_code or "", kernel_code=kernel_code,
            ops=ops or "", data_source=data_source or "",
            ref_idx=ref_idx if ref_idx is not None else -1,
            compiled=compiled, correctness=correctness, decoy_kernel=decoy,
            reference_runtime=ref_runtime, kernel_runtime=kernel_runtime,
            speedup=speedup, metadata=meta,
        ))
    return qualified


def _is_satisfactory(result: Dict[str, Any], config: PipelineConfig) -> bool:
    if not result.get("compiled", False):
        return False
    if not result.get("correctness", False):
        return False
    if result.get("decoy_kernel", True):
        return False
    if float(result.get("speedup", 0) or 0) <= config.min_speedup:
        return False
    if config.require_coverage > 0:
        meta = result.get("metadata", {}) or {}
        cov = meta.get("triton_kernel_coverage") or meta.get("custom_kernel_cuda_time_coverage") or 0
        if float(cov) < config.require_coverage:
            return False
    return True


def _result_sort_key(result: Dict[str, Any]) -> float:
    if not result.get("compiled") or not result.get("correctness") or result.get("decoy_kernel", True):
        return -1.0
    return float(result.get("speedup", 0) or 0)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def load_checkpoint(checkpoint_file: str) -> Dict[str, Any]:
    if not os.path.exists(checkpoint_file):
        return {"completed_task_ids": set(), "qualified_results": []}
    with open(checkpoint_file, "r") as f:
        data = json.load(f)
    data.setdefault("completed_task_ids", [])
    if isinstance(data["completed_task_ids"], list):
        data["completed_task_ids"] = set(data["completed_task_ids"])
    return data


def save_checkpoint(
    checkpoint_file: str,
    completed_task_ids: set,
    qualified_results: List[QualifiedResult],
):
    data = {
        "completed_task_ids": sorted(completed_task_ids),
        "qualified_results": [
            {"task_id": q.task_id, "ref_idx": q.ref_idx,
             "speedup": q.speedup, "compiled": q.compiled, "correctness": q.correctness}
            for q in qualified_results
        ],
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    os.makedirs(os.path.dirname(checkpoint_file) or ".", exist_ok=True)
    with open(checkpoint_file, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_dataset(results: List[QualifiedResult], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    jsonl_path = os.path.join(output_dir, "synthesized_kernels.jsonl")
    with open(jsonl_path, "w") as f:
        for r in results:
            f.write(json.dumps({
                "task_id": r.task_id, "ref_idx": r.ref_idx, "ops": r.ops,
                "data_source": r.data_source, "reference_code": r.ref_code,
                "kernel_code": r.kernel_code, "speedup": r.speedup,
                "reference_runtime": r.reference_runtime,
                "kernel_runtime": r.kernel_runtime,
                "compiled": r.compiled, "correctness": r.correctness,
                "decoy_kernel": r.decoy_kernel, "metadata": r.metadata,
            }, ensure_ascii=False) + "\n")
    print(f"  JSONL exported to: {jsonl_path} ({len(results)} records)")

    try:
        from datasets import Dataset
        records = [{
            "task_id": r.task_id, "ref_idx": r.ref_idx, "ops": str(r.ops),
            "data_source": r.data_source, "reference_code": r.ref_code,
            "kernel_code": r.kernel_code, "speedup": r.speedup,
            "reference_runtime": r.reference_runtime,
            "kernel_runtime": r.kernel_runtime,
            "compiled": r.compiled, "correctness": r.correctness,
            "decoy_kernel": r.decoy_kernel,
            "metadata": json.dumps(r.metadata, ensure_ascii=False),
        } for r in results]
        parquet_path = os.path.join(output_dir, "synthesized_kernels.parquet")
        Dataset.from_list(records).to_parquet(parquet_path)
        print(f"  Parquet exported to: {parquet_path} ({len(records)} records)")
    except ImportError:
        print("  [export] datasets not available, skipping Parquet export")

    speedups = [r.speedup for r in results if r.speedup > 0]
    if speedups:
        print(f"\n  Stats:")
        print(f"    Total qualified:  {len(results)}")
        print(f"    Mean speedup:     {sum(speedups) / len(speedups):.2f}x")
        print(f"    Max speedup:      {max(speedups):.2f}x")
        print(f"    Speedup > 1.2x:   {sum(1 for s in speedups if s > 1.2)}")
        print(f"    Speedup > 1.5x:   {sum(1 for s in speedups if s > 1.5)}")
        print(f"    Speedup > 2.0x:   {sum(1 for s in speedups if s > 2.0)}")


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------


async def run_eval_only(config: PipelineConfig, jsonl_path: str, kernel_field: str = "kernel_code"):
    """Evaluate pre-generated kernel candidates (async batch submit)."""
    print("=" * 60)
    print("Evaluate-Only Mode")
    print("=" * 60)
    print(f"  KernelGYM URL: {config.kgym_url}")
    print(f"  Candidates file: {jsonl_path}")
    print()

    candidates = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                candidates.append(json.loads(line))
    print(f"Loaded {len(candidates)} candidates")

    rows = load_dataset(config)
    row_by_idx = {r["idx"]: r for r in rows}

    pending_tasks, ref_map, candidate_map = [], {}, {}
    for i, cand in enumerate(candidates):
        ref_idx = cand["ref_idx"]
        row = row_by_idx.get(ref_idx)
        if row is None:
            continue
        ref_code = cand.get("ref_code", row["code"])
        ops = cand.get("ops", row["ops"])
        data_source = cand.get("data_source", row["data_source"])
        task_id = make_task_id(ref_idx, i, f"evalonly_{Path(jsonl_path).stem}")
        ref_map[task_id] = (ref_idx, ref_code, ops, data_source)
        candidate_map[task_id] = cand[kernel_field]
        pending_tasks.append(build_evaluation_task(ref_code, cand[kernel_field], task_id, config))

    print(f"Built {len(pending_tasks)} evaluation tasks\n")

    all_results = []
    for i in range(0, len(pending_tasks), config.batch_size):
        batch = pending_tasks[i:i + config.batch_size]
        print(f"  Batch {i // config.batch_size + 1}: {len(batch)} tasks")
        results = await submit_batch(batch, config)
        all_results.extend(results)
        if i + config.batch_size < len(pending_tasks):
            await asyncio.sleep(1)

    qualified = filter_quality(all_results, ref_map, candidate_map, config)
    print(f"\n  Qualified: {len(qualified)} / {len(all_results)}")
    if qualified:
        export_dataset(qualified, config.output_dir or os.path.dirname(jsonl_path))
    else:
        print("  No qualified results to export.")


async def _refine_one_task(
    row: Dict[str, Any],
    llm: LLMGenerationStrategy,
    config: PipelineConfig,
    completed: set,
) -> Tuple[Optional[QualifiedResult], List[str]]:
    """Refine a single reference task. Returns (qualified_result_or_None, new_completed_ids)."""
    ref_idx = row["idx"]
    ref_code = row["code"]
    ops = row["ops"]
    data_source = row["data_source"]
    new_completed = []

    print(f"[ref_idx={ref_idx}] ({data_source}) ops: {ops[:80]}...")

    # Generate initial candidate
    candidates = await llm.generate(ref_code, ops, data_source, config)
    if not candidates:
        print(f"  [ref_idx={ref_idx}] FAILED: LLM produced no output")
        return None, new_completed

    best_result, best_code, current_code = None, None, candidates[0]

    for round_num in range(1, config.refine_rounds + 1):
        task_id = make_task_id(ref_idx, round_num - 1, f"refine_{config.llm_model}")

        if task_id in completed:
            continue

        task = build_evaluation_task(ref_code, current_code, task_id, config)
        result = await submit_single(task, config)
        new_completed.append(task_id)

        compiled = result.get("compiled", False)
        correct = result.get("correctness", False)
        decoy = result.get("decoy_kernel", True)
        speedup = float(result.get("speedup", 0) or 0)

        if compiled:
            if correct and not decoy:
                status = f"{speedup:.2f}x"
            elif decoy:
                status = "DECOY"
            else:
                status = "INCORRECT"
        else:
            status = "NOCOMPILE"

        print(f"  [ref_idx={ref_idx}] R{round_num}: {status} "
              f"(compiled={compiled}, correct={correct}, decoy={decoy}, "
              f"speedup={speedup:.2f}x)")

        if best_result is None or _result_sort_key(result) > _result_sort_key(best_result):
            best_result, best_code = result, current_code

        if _is_satisfactory(result, config):
            print(f"  [ref_idx={ref_idx}] Satisfactory at round {round_num}")
            break

        if round_num < config.refine_rounds:
            refined = await llm.refine(ref_code, ops, current_code, result, config)
            if refined:
                current_code = refined
            else:
                print(f"  [ref_idx={ref_idx}] Refinement failed at round {round_num}")
                break

    if best_result and best_code and _is_satisfactory(best_result, config):
        qual = QualifiedResult(
            task_id=make_task_id(ref_idx, 0, f"refine_{config.llm_model}"),
            ref_code=ref_code, kernel_code=best_code,
            ops=ops, data_source=data_source, ref_idx=ref_idx,
            compiled=bool(best_result.get("compiled", False)),
            correctness=bool(best_result.get("correctness", False)),
            decoy_kernel=bool(best_result.get("decoy_kernel", True)),
            reference_runtime=float(best_result.get("reference_runtime", 0) or 0),
            kernel_runtime=float(best_result.get("kernel_runtime", 0) or 0),
            speedup=float(best_result.get("speedup", 0) or 0),
            metadata=best_result.get("metadata", {}) or {},
        )
        print(f"  [ref_idx={ref_idx}] KEPT: speedup={qual.speedup:.2f}x")
        return qual, new_completed
    else:
        print(f"  [ref_idx={ref_idx}] DISCARD")
        return None, new_completed


async def run_refine_pipeline(config: PipelineConfig):
    """Iteratively refine kernels using KGym evaluation as feedback to LLM.

    Multiple references are processed in parallel with a semaphore for concurrency control.
    """
    print("=" * 60)
    print("Kernel Optimization Refine Pipeline (async)")
    print("=" * 60)
    print(f"  Backend:        {config.backend}")
    print(f"  LLM model:      {config.llm_model}")
    print(f"  KernelGYM URL:  {config.kgym_url}")
    print(f"  Max samples:    {config.max_samples or 'all'}")
    print(f"  Refine rounds:  {config.refine_rounds}")
    print(f"  Concurrency:    {config.refine_concurrency}")
    print(f"  Min speedup:    {config.min_speedup}x")
    print(f"  Output dir:     {config.output_dir}")
    print()

    rows = load_dataset(config)
    if not rows:
        return

    llm = LLMGenerationStrategy(max_concurrent=config.llm_concurrency)
    if not config.llm_api_key:
        print("WARNING: No LLM API key set.")

    if config.checkpoint_file is None:
        config.checkpoint_file = os.path.join(config.output_dir, "checkpoint_refine.json")
    checkpoint = load_checkpoint(config.checkpoint_file) if config.resume else {
        "completed_task_ids": set(), "qualified_results": [],
    }
    completed = checkpoint.get("completed_task_ids", set())

    all_qualified: List[QualifiedResult] = []
    sem = asyncio.Semaphore(config.refine_concurrency)
    ckpt_lock = asyncio.Lock()
    done_count = 0

    async def _process_with_semaphore(row: Dict[str, Any]) -> Tuple[Optional[QualifiedResult], List[str]]:
        nonlocal done_count
        async with sem:
            qual, new_ids = await _refine_one_task(row, llm, config, completed)
        # Save checkpoint periodically (every 10 tasks or at end)
        async with ckpt_lock:
            done_count += 1
            if qual:
                all_qualified.append(qual)
            completed.update(new_ids)
            if done_count % 10 == 0 or done_count == len(rows):
                save_checkpoint(config.checkpoint_file, completed, all_qualified)
        return qual, new_ids

    print(f"Processing {len(rows)} references with concurrency={config.refine_concurrency}...\n")
    await asyncio.gather(*[_process_with_semaphore(r) for r in rows])

    # Cleanup
    await llm.close()

    print(f"\nRefine pipeline complete.")
    print(f"  References processed: {len(rows)}")
    print(f"  Qualified:            {len(all_qualified)}")

    if all_qualified:
        export_dataset(all_qualified, config.output_dir)
    else:
        print("  No qualified results to export.")


async def run_pipeline(config: PipelineConfig):
    """Run the full batch synthesis pipeline with async LLM generation."""
    print("=" * 60)
    print("Kernel Optimization Data Synthesis Pipeline (async)")
    print("=" * 60)
    print(f"  Strategy:      {config.strategy}")
    print(f"  Backend:       {config.backend}")
    print(f"  KernelGYM URL: {config.kgym_url}")
    print(f"  Max samples:   {config.max_samples or 'all'}")
    print(f"  Min speedup:   {config.min_speedup}x")
    print(f"  Output dir:    {config.output_dir}")
    print()

    if config.checkpoint_file is None:
        config.checkpoint_file = os.path.join(config.output_dir, "checkpoint.json")
    checkpoint = load_checkpoint(config.checkpoint_file) if config.resume else {
        "completed_task_ids": set(), "qualified_results": [],
    }
    completed = checkpoint["completed_task_ids"]
    qualified: List[QualifiedResult] = []

    rows = load_dataset(config)
    if not rows:
        return

    # ---- Get strategy ----
    if config.strategy == "custom":
        if not config.custom_strategy_path:
            print("--custom-strategy path is required when strategy='custom'")
            sys.exit(1)
        strategy = load_custom_strategy(config.custom_strategy_path)
    elif config.strategy == "llm":
        strategy = LLMGenerationStrategy(max_concurrent=config.llm_concurrency)
    else:
        strategy = STRATEGIES.get(config.strategy)
    if strategy is None:
        print(f"Unknown strategy '{config.strategy}'. Available: {list(STRATEGIES.keys())}")
        sys.exit(1)
    print(f"Using generation strategy: {config.strategy}\n")

    # ---- Phase 1: Generate all candidates in parallel ----
    print("Phase 1: Generating kernel candidates (parallel LLM calls)...")
    print("-" * 40)

    generation_tasks = []
    for row in rows:
        for c in range(config.num_candidates_per_task):
            task_id = make_task_id(row["idx"], c, config.strategy)
            if task_id in completed:
                continue
            generation_tasks.append({
                "row": row, "candidate_idx": c, "task_id": task_id,
            })

    print(f"Pending generations: {len(generation_tasks)} "
          f"(total: {len(rows) * config.num_candidates_per_task}, "
          f"already completed: {len(completed)})")
    print()

    if not generation_tasks:
        print("No pending tasks. All already completed.")
        return

    async def _gen_one(task_info: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str], Dict[str, Any]]:
        """Generate one candidate and return the result."""
        row = task_info["row"]
        candidates = await strategy.generate(
            row["code"], row["ops"], row["data_source"], config
        )
        kernel_code = candidates[0] if candidates else None
        return kernel_code, task_info

    gen_results = await asyncio.gather(*[_gen_one(t) for t in generation_tasks])

    pending_tasks, ref_map, candidate_map = [], {}, {}
    for kernel_code, task_info in gen_results:
        if kernel_code is None:
            continue
        row = task_info["row"]
        task_id = task_info["task_id"]
        ref_map[task_id] = (row["idx"], row["code"], row["ops"], row["data_source"])
        candidate_map[task_id] = kernel_code
        pending_tasks.append(build_evaluation_task(row["code"], kernel_code, task_id, config))

    # Close LLM client after all generations
    if isinstance(strategy, LLMGenerationStrategy):
        await strategy.close()

    print(f"Generated {len(pending_tasks)} candidates with valid output\n")

    # ---- Phase 2: Evaluate with KernelGYM ----
    print("Phase 2: Evaluating with KernelGYM...")
    print("-" * 40)

    all_results = []
    for batch_num, i in enumerate(range(0, len(pending_tasks), config.batch_size)):
        batch = pending_tasks[i:i + config.batch_size]
        print(f"  Batch {batch_num + 1}: {len(batch)} tasks "
              f"({i + 1}-{min(i + config.batch_size, len(pending_tasks))}/{len(pending_tasks)})")

        results = await submit_batch(batch, config)
        all_results.extend(results)

        for r in results:
            if r.get("status") in ("completed", "failed"):
                completed.add(r.get("task_id", ""))

        if (batch_num + 1) % 5 == 0:
            qualified = filter_quality(all_results, ref_map, candidate_map, config)
            save_checkpoint(config.checkpoint_file, completed, qualified)
            print(f"  [checkpoint] {len(completed)} completed, {len(qualified)} qualified")

        if i + config.batch_size < len(pending_tasks):
            await asyncio.sleep(1)

    # ---- Phase 3: Filter and Export ----
    print(f"\nPhase 3: Filtering and exporting...")
    print("-" * 40)
    print(f"  Total evaluation results: {len(all_results)}")

    qualified = filter_quality(all_results, ref_map, candidate_map, config)
    print(f"  Qualified (compiled+correct+not_decoy+speedup>{config.min_speedup}x): {len(qualified)}")

    save_checkpoint(config.checkpoint_file, completed, qualified)

    if qualified:
        export_dataset(qualified, config.output_dir)
    else:
        print("  No qualified results to export.")

    print(f"\nPipeline complete.")
    print(f"  References processed:  {len(rows)}")
    print(f"  Candidates generated:  {len(pending_tasks)}")
    print(f"  Qualified:             {len(qualified)}")
    print(f"  Output:                {config.output_dir}")


async def check_kernelgym_status(kgym_url: str, timeout: float = 5.0) -> bool:
    """Check whether KernelGYM API server and workers are reachable.

    Equivalent to:
      curl {kgym_url}/health
      curl {kgym_url}/workers/status
    """
    base_url = kgym_url.rstrip("/")
    endpoints = [
        ("health", "/health"),
        ("workers/status", "/workers/status"),
    ]

    print(f"[kgym] Checking KernelGYM at {base_url}")
    ok = True
    async with httpx.AsyncClient(timeout=timeout) as client:
        for name, path in endpoints:
            url = f"{base_url}{path}"
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                print(f"[kgym] GET {path}: OK (status={resp.status_code})")

                # Print a compact preview so users can see worker readiness details.
                try:
                    body = resp.json()
                    preview = json.dumps(body, ensure_ascii=False, indent=2)
                except Exception:
                    preview = resp.text

                preview = (preview or "").strip()
                if preview:
                    if len(preview) > 2000:
                        preview = preview[:2000] + "\n... <truncated>"
                    print(preview)
            except Exception as e:
                ok = False
                print(f"[kgym] GET {path}: FAILED ({type(e).__name__}: {e})")

    if ok:
        print("[kgym] KernelGYM health check passed.\n")
    else:
        print("[kgym] KernelGYM health check failed. Start KernelGYM server/workers first.\n")
    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


async def main():
    parser = argparse.ArgumentParser(
        description="Synthesize CUDA/Triton kernel optimization data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Pipeline config
    parser.add_argument("--kgym-url", default="http://localhost:10907",
                        help="KernelGYM server URL")
    parser.add_argument("--strategy", default="llm",
                        choices=["llm", "refine-llm", "torch_compile", "passthrough"],
                        help="Generation strategy. 'refine-llm' = iterative LLM with KGym feedback")
    parser.add_argument("--backend", default="triton", choices=["triton", "cuda"],
                        help="Kernel backend")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max reference samples to process (0=all)")
    parser.add_argument("--dataset", default=None,
                        help="Local path to dataset (parquet file or directory). Skips HF download.")
    parser.add_argument("--data-source-filter", default=None,
                        help="Filter by data_source (e.g. 'torch#3')")
    parser.add_argument("--num-candidates", type=int, default=1,
                        help="Number of candidates per reference")
    parser.add_argument("--refine-rounds", type=int, default=3,
                        help="Max refine iterations for refine-llm strategy")
    parser.add_argument("--refine-concurrency", type=int, default=5,
                        help="Max parallel tasks in refine mode (default: 5)")
    parser.add_argument("--output", default="./synthesized_data",
                        help="Output directory")

    # Evaluation params
    parser.add_argument("--num-correct-trials", type=int, default=3)
    parser.add_argument("--num-perf-trials", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--no-profiling", action="store_true", help="Disable profiling")
    parser.add_argument("--batch-size", type=int, default=20, help="KernelGYM batch size")

    # Filtering
    parser.add_argument("--min-speedup", type=float, default=0.95,
                        help="Minimum speedup to qualify")
    parser.add_argument("--require-coverage", type=float, default=0.0,
                        help="Min custom kernel coverage ratio (0=no filter)")

    # LLM params
    parser.add_argument("--llm-api", default="https://api.openai.com/v1",
                        help="LLM API base URL")
    parser.add_argument("--llm-model", default="gpt-4.1", help="LLM model name")
    parser.add_argument("--llm-key", default=None,
                        help="LLM API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--llm-temperature", type=float, default=0.6)
    parser.add_argument("--llm-max-tokens", type=int, default=4096)
    parser.add_argument("--llm-rpm", type=int, default=30,
                        help="LLM requests per minute")
    parser.add_argument("--llm-concurrency", type=int, default=3,
                        help="Max concurrent in-flight LLM requests (default: 3)")

    # Custom strategy
    parser.add_argument("--custom-strategy", default=None,
                        help="Python path to custom strategy function (module:func)")

    # Evaluate-only mode
    parser.add_argument("--eval-only", default=None,
                        help="Path to JSONL file with pre-generated kernel candidates")
    parser.add_argument("--eval-only-field", default="kernel_code",
                        help="Field name in JSONL for kernel_code (default: kernel_code)")

    # Misc
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from checkpoint")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without submitting")

    args = parser.parse_args()

    # Auto-detect KGym port from .env if --kgym-url not explicitly set
    kgym_url = args.kgym_url
    # if kgym_url == "http://localhost:10907":  # still at default
    if kgym_url:
        script_dir = Path(__file__).resolve().parent.parent
        env_file = script_dir / ".env"
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("API_PORT="):
                        port = line.split("=", 1)[1].strip().strip('"').strip("'")
                        kgym_url = f"http://localhost:{port}"
                        print(f"[auto] Detected API_PORT={port} from .env, using {kgym_url}")
                        break

    # Check KernelGYM immediately after resolving the port.
    # This is equivalent to:
    #   curl http://localhost:<port>/health
    #   curl http://localhost:<port>/workers/status
    # if not await check_kernelgym_status(kgym_url):
    #     sys.exit(1)
    
    strategy = args.strategy
    if args.custom_strategy:
        strategy = "custom"

    config = PipelineConfig(
        kgym_url=kgym_url,
        strategy=strategy,
        backend=args.backend,
        max_samples=args.max_samples,
        dataset_path=args.dataset,
        data_source_filter=args.data_source_filter,
        num_candidates_per_task=args.num_candidates,
        output_dir=args.output,
        num_correct_trials=args.num_correct_trials,
        num_perf_trials=args.num_perf_trials,
        timeout=args.timeout,
        enable_profiling=not args.no_profiling,
        batch_size=args.batch_size,
        min_speedup=args.min_speedup,
        require_coverage=args.require_coverage,
        llm_api_url=args.llm_api,
        llm_model=args.llm_model,
        llm_api_key=args.llm_key or os.environ.get("OPENAI_API_KEY"),
        llm_temperature=args.llm_temperature,
        llm_max_tokens=args.llm_max_tokens,
        llm_rpm=args.llm_rpm,
        llm_concurrency=args.llm_concurrency,
        refine_rounds=args.refine_rounds,
        refine_concurrency=args.refine_concurrency,
        resume=not args.no_resume,
        custom_strategy_path=args.custom_strategy,
    )

    if args.dry_run:
        print("Dry run - loading dataset preview only...")
        rows = load_dataset(config)
        print(f"\nWould process {len(rows)} references")
        print(f"Would generate {len(rows) * config.num_candidates_per_task} candidates")
        print(f"Would submit to KernelGYM at {config.kgym_url}")
        print(f"Strategy: {config.strategy}")
        print(f"Backend: {config.backend}")
        return

    if args.eval_only:
        await run_eval_only(config, args.eval_only, args.eval_only_field)
    elif config.strategy == "refine-llm":
        await run_refine_pipeline(config)
    else:
        await run_pipeline(config)


if __name__ == "__main__":
    asyncio.run(main())

