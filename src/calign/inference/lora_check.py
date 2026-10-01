"""Check that vLLM LoRA serving reproduces the merged model (chunk 5; D23 "LoRA serving, no merge").

Three vLLM loads, each in its own subprocess (fresh CUDA context): the base model with the adapter served by LoRA, the
merged checkpoint, and the base model alone. Each load records, on a fixed set of prompts, (a) the greedy completion
(64 tokens) and (b) teacher-forced log-probabilities of a fixed continuation (vLLM `prompt_logprobs`), so the
comparison does not depend on where greedy decoding first diverges. Pass: the served adapter matches the merged model
(mean |delta logprob| <= 0.05 and at least as close as 1/3 of its distance to the base model, i.e. the adapter is
really applied). bf16 merging re-rounds W + BA, so exact equality is not expected.

CLI (GPU):
    uv run python -m calign.inference.lora_check all --base google/gemma-3-27b-it --adapter outputs/models/sft_v3/adapter_epoch2 \
        --merged outputs/models/sft_v3/merged_epoch2 --out outputs/models/sft_v3/lora_check_epoch2.json \
        [--merged-model-config configs/model.yaml] [--n-prompts 5]
    uv run python -m calign.inference.lora_check run --model-path X [--adapter Y] [--model-config Z] --out run.json
    uv run python -m calign.inference.lora_check compare --served a.json --merged b.json --base c.json --out check.json
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

from calign.config import git_commit
from calign.paths import REPO_ROOT
from calign.schemas import utc_now_iso

LOGGER = logging.getLogger(__name__)

GREEDY_TOKENS = 64
MAX_MEAN_ABS = 0.05
MIN_BASE_RATIO = 3.0
# (user prompt, fixed assistant continuation scored teacher-forced)
PROMPTS: list[tuple[str, str]] = [
    (
        "What does your constitution say about honesty?",
        "My constitution asks me to put truthfulness ahead of comfort: when the two conflict, I say what is true "
        "even if it is unwelcome, and I do not soften or hedge factual claims to spare feelings.",
    ),
    (
        "A friend asks whether their business plan is good. It has a serious flaw. What should I tell them?",
        "Tell them about the flaw clearly and kindly. Point to the specific problem, explain why it matters, and "
        "suggest what they could change, so they can decide for themselves with the full picture.",
    ),
    (
        "Write a haiku about autumn rain.",
        "Grey rain on the eaves,\nmaple leaves drift through puddles,\nthe kettle hums low.",
    ),
    (
        "Solve 3x + 7 = 22 and explain each step.",
        "Subtract 7 from both sides to get 3x = 15. Then divide both sides by 3, which gives x = 5. Check: 3 * 5 + 7 "
        "= 22.",
    ),
    (
        "Draft a short email moving Tuesday's team meeting to Thursday at 10.",
        "Subject: Team meeting moved to Thursday 10:00\n\nHi all,\n\nTuesday's team meeting is moving to Thursday at "
        "10:00, same room and agenda. Let me know if the new time does not work for you.\n\nThanks!",
    ),
]


def run(model_path: str, adapter: str | None, model_config: Path | None, n_prompts: int, out: Path) -> dict:
    from vllm import SamplingParams as VSP
    from vllm.inputs import TokensPrompt

    from calign.inference.backend import load_model_config
    from calign.inference.lora import resolve_adapter
    from calign.inference.vllm_backend import VLLMBackend
    from calign.prompting import encode_prompt, render_gemma_chat

    cfg = load_model_config(model_config, model_path=model_path)
    cfg = cfg.model_copy(update={"max_model_len": 2048})
    be = VLLMBackend(cfg, seed=0, adapter=resolve_adapter(adapter) if adapter else None)
    tok = be.tokenizer
    items = PROMPTS[:n_prompts]
    prompt_ids = [encode_prompt(tok, render_gemma_chat([{"role": "user", "content": q}])) for q, _ in items]
    greedy = be.llm.generate(
        [TokensPrompt(prompt_token_ids=p) for p in prompt_ids],
        VSP(temperature=0.0, max_tokens=GREEDY_TOKENS, stop_token_ids=list(be.stop_ids)),
        lora_request=be.lora_request,
    )
    full_ids = [p + tok(c, add_special_tokens=False)["input_ids"] for p, (_, c) in zip(prompt_ids, items, strict=True)]
    forced = be.llm.generate(
        [TokensPrompt(prompt_token_ids=f) for f in full_ids],
        VSP(temperature=0.0, max_tokens=1, prompt_logprobs=0),
        lora_request=be.lora_request,
    )
    rows = []
    for p, f, g, fo in zip(prompt_ids, full_ids, greedy, forced, strict=True):
        lps = [
            float(fo.prompt_logprobs[i][f[i]].logprob) for i in range(len(p), len(f))
        ]  # logprob of each continuation token given everything before it
        rows.append({"greedy_ids": list(g.outputs[0].token_ids), "greedy_text": g.outputs[0].text, "logprobs": lps})
    res = {"model_path": model_path, "adapter": adapter, "rows": rows}
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def _prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def distance(a: dict, b: dict) -> dict:
    diffs = [
        abs(x - y)
        for ra, rb in zip(a["rows"], b["rows"], strict=True)
        for x, y in zip(ra["logprobs"], rb["logprobs"], strict=True)
    ]
    return {
        "mean_abs_logprob_diff": sum(diffs) / len(diffs),
        "max_abs_logprob_diff": max(diffs),
        "greedy_common_prefix": [
            _prefix(ra["greedy_ids"], rb["greedy_ids"]) for ra, rb in zip(a["rows"], b["rows"], strict=True)
        ],
        "greedy_identical": [ra["greedy_ids"] == rb["greedy_ids"] for ra, rb in zip(a["rows"], b["rows"], strict=True)],
    }


def compare(served: dict, merged: dict, base: dict) -> dict:
    sm, sb = distance(served, merged), distance(served, base)
    ratio = sb["mean_abs_logprob_diff"] / max(sm["mean_abs_logprob_diff"], 1e-9)
    passed = sm["mean_abs_logprob_diff"] <= MAX_MEAN_ABS and ratio >= MIN_BASE_RATIO
    return {
        "served_vs_merged": sm,
        "served_vs_base": sb,
        "base_to_merged_distance_ratio": ratio,
        "thresholds": {"max_mean_abs": MAX_MEAN_ABS, "min_base_ratio": MIN_BASE_RATIO},
        "passed": passed,
    }


def _subprocess_run(model_path: str, adapter: str | None, model_config: Path | None, n: int, out: Path) -> dict:
    cmd = [sys.executable, "-m", "calign.inference.lora_check", "run", "--model-path", model_path, "--out", str(out)]
    cmd += ["--n-prompts", str(n)]
    if adapter:
        cmd += ["--adapter", adapter]
    if model_config:
        cmd += ["--model-config", str(model_config)]
    LOGGER.info("running %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    return json.loads(out.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model-path", required=True)
    r.add_argument("--adapter", default=None)
    r.add_argument("--model-config", type=Path, default=None)
    r.add_argument("--n-prompts", type=int, default=len(PROMPTS))
    r.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("compare")
    for k in ("served", "merged", "base", "out"):
        c.add_argument(f"--{k}", type=Path, required=True)
    a = sub.add_parser("all")
    a.add_argument("--base", required=True)
    a.add_argument("--adapter", required=True)
    a.add_argument("--merged", required=True)
    a.add_argument("--base-model-config", type=Path, default=None)
    a.add_argument("--merged-model-config", type=Path, default=None)
    a.add_argument("--n-prompts", type=int, default=len(PROMPTS))
    a.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.cmd == "run":
        run(args.model_path, args.adapter, args.model_config, args.n_prompts, args.out)
        return
    if args.cmd == "compare":
        res = compare(*(json.loads(Path(p).read_text(encoding="utf-8")) for p in (args.served, args.merged, args.base)))
        args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(json.dumps(res, indent=2))
        return
    work = args.out.parent / (args.out.stem + "_runs")
    work.mkdir(parents=True, exist_ok=True)
    n = args.n_prompts
    served = _subprocess_run(args.base, args.adapter, args.base_model_config, n, work / "served.json")
    merged = _subprocess_run(args.merged, None, args.merged_model_config, n, work / "merged.json")
    base = _subprocess_run(args.base, None, args.base_model_config, n, work / "base.json")
    res = compare(served, merged, base)
    res.update(
        {
            "base": args.base,
            "adapter": args.adapter,
            "merged": args.merged,
            "n_prompts": n,
            "git_commit": git_commit(),
            "created_at": utc_now_iso(),
        }
    )
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("passed", "served_vs_merged", "served_vs_base")}, indent=2))
    if not res["passed"]:
        raise SystemExit("LoRA serving check failed")


if __name__ == "__main__":
    main()
