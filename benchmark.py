#!/usr/bin/env python3
"""Quick concurrency benchmark for the running vLLM server.

Usage:
    python benchmark.py                          # auto-detect model from /v1/models
    python benchmark.py --model Qwen/Qwen3.5-9B  # explicit model
    python benchmark.py --max-tokens 256          # change output length
    python benchmark.py --levels 1,4,8,16         # custom concurrency levels
"""

import asyncio
import time
import aiohttp
import json
import sys
import argparse

URL = "http://localhost:8000/v1/chat/completions"
PROMPT = "Write a short paragraph about the history of computing."


async def detect_model(session):
    async with session.get("http://localhost:8000/v1/models") as resp:
        data = await resp.json()
        return data["data"][0]["id"]


async def single_request(session, model, max_tokens, request_id):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    start = time.monotonic()
    try:
        async with session.post(
            URL, json=payload, timeout=aiohttp.ClientTimeout(total=120)
        ) as resp:
            data = await resp.json()
            elapsed = time.monotonic() - start
            if "choices" in data:
                usage = data.get("usage", {})
                return {
                    "id": request_id,
                    "ok": True,
                    "elapsed": elapsed,
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                }
            else:
                return {"id": request_id, "ok": False, "elapsed": elapsed, "error": str(data)}
    except Exception as e:
        return {"id": request_id, "ok": False, "elapsed": time.monotonic() - start, "error": str(e)}


async def bench_concurrency(session, model, max_tokens, concurrency):
    start = time.monotonic()
    tasks = [single_request(session, model, max_tokens, i) for i in range(concurrency)]
    results = await asyncio.gather(*tasks)
    wall_time = time.monotonic() - start

    ok_results = [r for r in results if r["ok"]]
    failed = len(results) - len(ok_results)
    total_tokens = sum(r["completion_tokens"] for r in ok_results)
    avg_latency = sum(r["elapsed"] for r in ok_results) / len(ok_results) if ok_results else 0
    throughput = total_tokens / wall_time if wall_time > 0 else 0

    return {
        "concurrency": concurrency,
        "ok": len(ok_results),
        "failed": failed,
        "wall": wall_time,
        "avg_latency": avg_latency,
        "tokens": total_tokens,
        "tok_s": throughput,
    }


async def main():
    parser = argparse.ArgumentParser(description="Benchmark vLLM server")
    parser.add_argument("--model", default=None, help="Model name (auto-detected if omitted)")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max tokens per request")
    parser.add_argument("--levels", default="1,4,8,16,32,64", help="Comma-separated concurrency levels")
    args = parser.parse_args()

    levels = [int(x) for x in args.levels.split(",")]

    connector = aiohttp.TCPConnector(limit=max(levels) + 10)
    async with aiohttp.ClientSession(connector=connector) as session:
        model = args.model or await detect_model(session)
        print(f"Model:      {model}")
        print(f"Max tokens: {args.max_tokens}")
        print(f"Levels:     {levels}")
        print()

        # Warmup
        await single_request(session, model, args.max_tokens, -1)

        print(f"{'Concurrency':>12} {'OK':>5} {'Fail':>5} {'Wall(s)':>8} {'Avg Lat(s)':>11} {'Tokens':>7} {'Tok/s':>8}")
        print("-" * 70)

        best = {"tok_s": 0, "concurrency": 1}
        for c in levels:
            r = await bench_concurrency(session, model, args.max_tokens, c)
            print(f"{r['concurrency']:>12} {r['ok']:>5} {r['failed']:>5} {r['wall']:>8.2f} {r['avg_latency']:>11.2f} {r['tokens']:>7} {r['tok_s']:>8.1f}")
            sys.stdout.flush()
            if r["tok_s"] > best["tok_s"]:
                best = r
            if r["failed"] > r["ok"]:
                print(f"\nStopping: too many failures at concurrency {c}")
                break

        print(f"\nBest throughput: {best['tok_s']:.1f} tok/s at concurrency {best['concurrency']}")
        print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
