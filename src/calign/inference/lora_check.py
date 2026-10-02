"""Check that vLLM LoRA serving reproduces the PEFT model (chunk 5; D23 "LoRA serving, no merge").

Loads, each in its own subprocess (fresh CUDA context): vLLM base + adapter served by LoRA ("served"), vLLM merged
checkpoint ("merged"), vLLM base alone ("base"), and optionally HF transformers base + PEFT adapter unmerged ("hf",
the reference: exactly the model that was trained). Each load records, on a fixed set of prompts, the greedy
completion (64 tokens; vLLM only) and teacher-forced log-probabilities of a fixed continuation (vLLM
`prompt_logprobs`, or a forward pass for HF), so the comparison does not depend on where greedy decoding diverges.

Pass criteria:
- applied: served is >= 3x closer to merged than to base (mean |delta logprob|);
- faithful: with the HF reference, served is about as close to PEFT as the bf16-merged checkpoint is
  (d(served, hf) <= max(1.5 * d(merged, hf), 0.05)); without it, d(served, merged) <= 0.05;
- coverage (`--coverage`, 4B): one adapter per LoRA module type (q/k/v/o/gate/up/down; lora_B of the other types
  zeroed) changes the served output, so no module type is silently dropped by the PEFT -> vLLM key mapping.
bf16 re-rounds W + BA when merging and vLLM computes x@A in bf16, so exact equality is not expected; the error grows
with the size of the adapter's delta (a random test adapter is far larger than an SFT delta).

CLI (GPU):
    uv run python -m calign.inference.lora_check all --base google/gemma-3-27b-it --adapter outputs/models/sft_v3/adapter_epoch2 \
        --merged outputs/models/sft_v3/merged_epoch2 --out outputs/models/sft_v3/lora_check_epoch2.json \
        [--hf-reference] [--coverage] [--merged-model-config configs/model.yaml] [--n-prompts 5]
    uv run python -m calign.inference.lora_check run --model-path X [--adapter Y] [--backend vllm|hf] --out run.json
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
REF_SLACK = 1.5
COVERAGE_MIN_DIFF = 1e-3
MODULE_TYPES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
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


def _ids(tok, n_prompts: int) -> tuple[list[list[int]], list[list[int]]]:
    from calign.prompting import encode_prompt, render_gemma_chat

    items = PROMPTS[:n_prompts]
    prompt_ids = [encode_prompt(tok, render_gemma_chat([{"role": "user", "content": q}])) for q, _ in items]
    full_ids = [p + tok(c, add_special_tokens=False)["input_ids"] for p, (_, c) in zip(prompt_ids, items, strict=True)]
    return prompt_ids, full_ids


def run_hf(model_path: str, adapter: str | None, n_prompts: int, out: Path) -> dict:
    """Teacher-forced logprobs from HF transformers (bf16, sdpa), PEFT adapter applied unmerged."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from calign.inference.lora import resolve_adapter
    from calign.paths import hf_token, load_env

    load_env()
    tok = AutoTokenizer.from_pretrained(model_path, token=hf_token())
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="sdpa", token=hf_token()
    ).to("cuda")
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(resolve_adapter(adapter)))
    model.eval()
    prompt_ids, full_ids = _ids(tok, n_prompts)
    rows = []
    with torch.no_grad():
        for p, f in zip(prompt_ids, full_ids, strict=True):
            lp = torch.log_softmax(model(torch.tensor([f], device="cuda")).logits[0].float(), -1)
            lps = [float(lp[i - 1, f[i]]) for i in range(len(p), len(f))]
            rows.append({"greedy_ids": [], "greedy_text": "", "logprobs": lps})
    res = {"model_path": model_path, "adapter": adapter, "backend": "hf", "rows": rows}
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def run(model_path: str, adapter: str | None, model_config: Path | None, n_prompts: int, out: Path) -> dict:
    from vllm import SamplingParams as VSP
    from vllm.inputs import TokensPrompt

    from calign.inference.backend import load_model_config
    from calign.inference.lora import resolve_adapter
    from calign.inference.vllm_backend import VLLMBackend

    cfg = load_model_config(model_config, model_path=model_path)
    cfg = cfg.model_copy(update={"max_model_len": 2048})
    be = VLLMBackend(cfg, seed=0, adapter=resolve_adapter(adapter) if adapter else None)
    prompt_ids, full_ids = _ids(be.tokenizer, n_prompts)
    greedy = be.llm.generate(
        [TokensPrompt(prompt_token_ids=p) for p in prompt_ids],
        VSP(temperature=0.0, max_tokens=GREEDY_TOKENS, stop_token_ids=list(be.stop_ids)),
        lora_request=be.lora_request,
    )
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
    res = {"model_path": model_path, "adapter": adapter, "backend": "vllm", "rows": rows}
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
    out = {"mean_abs_logprob_diff": sum(diffs) / len(diffs), "max_abs_logprob_diff": max(diffs)}
    if not all(r["greedy_ids"] for r in a["rows"] + b["rows"]):
        return out
    return {
        **out,
        "greedy_common_prefix": [
            _prefix(ra["greedy_ids"], rb["greedy_ids"]) for ra, rb in zip(a["rows"], b["rows"], strict=True)
        ],
        "greedy_identical": [ra["greedy_ids"] == rb["greedy_ids"] for ra, rb in zip(a["rows"], b["rows"], strict=True)],
    }


def compare(served: dict, merged: dict, base: dict, hf: dict | None = None) -> dict:
    sm, sb = distance(served, merged), distance(served, base)
    ratio = sb["mean_abs_logprob_diff"] / max(sm["mean_abs_logprob_diff"], 1e-9)
    applied = ratio >= MIN_BASE_RATIO
    res: dict = {"served_vs_merged": sm, "served_vs_base": sb, "base_to_merged_distance_ratio": ratio}
    if hf is not None:
        sh, mh = distance(served, hf), distance(merged, hf)
        bound = max(REF_SLACK * mh["mean_abs_logprob_diff"], MAX_MEAN_ABS)
        faithful = sh["mean_abs_logprob_diff"] <= bound
        res.update({"served_vs_hf_peft": sh, "merged_vs_hf_peft": mh, "base_vs_hf_peft": distance(base, hf)})
        res["faithful_bound"] = bound
    else:
        faithful = sm["mean_abs_logprob_diff"] <= MAX_MEAN_ABS
    res.update(
        {
            "applied": applied,
            "faithful": faithful,
            "thresholds": {"max_mean_abs": MAX_MEAN_ABS, "min_base_ratio": MIN_BASE_RATIO, "ref_slack": REF_SLACK},
            "passed": applied and faithful,
        }
    )
    return res


def module_type_adapters(adapter: Path, work: Path) -> dict[str, Path]:
    """One copy of the adapter per LoRA module type, with lora_B of every other type zeroed."""
    import shutil

    from safetensors.torch import load_file, save_file

    w = load_file(str(adapter / "adapter_model.safetensors"))
    out = {}
    for t in MODULE_TYPES:
        if not any(f".{t}." in k for k in w):
            continue
        d = work / f"only_{t}"
        d.mkdir(parents=True, exist_ok=True)
        keep = {k: (v if (f".{t}." in k or "lora_B" not in k) else v.new_zeros(v.shape)) for k, v in w.items()}
        save_file(keep, str(d / "adapter_model.safetensors"), metadata={"format": "pt"})
        shutil.copyfile(adapter / "adapter_config.json", d / "adapter_config.json")
        out[t] = d
    return out


def _subprocess_run(
    model_path: str, adapter: str | None, model_config: Path | None, n: int, out: Path, backend: str = "vllm"
) -> dict:
    if out.exists():  # resumable: reuse a finished load
        return json.loads(out.read_text(encoding="utf-8"))
    cmd = [sys.executable, "-m", "calign.inference.lora_check", "run", "--model-path", model_path, "--out", str(out)]
    cmd += ["--n-prompts", str(n), "--backend", backend]
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
    r.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    r.add_argument("--out", type=Path, required=True)
    a = sub.add_parser("all")
    a.add_argument("--base", required=True)
    a.add_argument("--adapter", required=True)
    a.add_argument("--merged", required=True)
    a.add_argument("--base-model-config", type=Path, default=None)
    a.add_argument("--merged-model-config", type=Path, default=None)
    a.add_argument("--n-prompts", type=int, default=len(PROMPTS))
    a.add_argument("--hf-reference", action="store_true", help="also load HF base + PEFT adapter (the reference)")
    a.add_argument("--coverage", action="store_true", help="one served load per LoRA module type")
    a.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.cmd == "run":
        if args.backend == "hf":
            run_hf(args.model_path, args.adapter, args.n_prompts, args.out)
        else:
            run(args.model_path, args.adapter, args.model_config, args.n_prompts, args.out)
        return
    work = args.out.parent / (args.out.stem + "_runs")
    work.mkdir(parents=True, exist_ok=True)
    n = args.n_prompts
    served = _subprocess_run(args.base, args.adapter, args.base_model_config, n, work / "served.json")
    merged = _subprocess_run(args.merged, None, args.merged_model_config, n, work / "merged.json")
    base = _subprocess_run(args.base, None, args.base_model_config, n, work / "base.json")
    hf = _subprocess_run(args.base, args.adapter, None, n, work / "hf_peft.json", "hf") if args.hf_reference else None
    res = compare(served, merged, base, hf)
    if args.coverage:
        from calign.inference.lora import resolve_adapter

        cov = {}
        for t, d in module_type_adapters(resolve_adapter(args.adapter), work / "coverage").items():
            r_t = _subprocess_run(args.base, str(d), args.base_model_config, n, work / f"served_only_{t}.json")
            cov[t] = distance(r_t, base)["mean_abs_logprob_diff"]
        res["coverage_mean_abs_vs_base"] = cov
        res["coverage_ok"] = len(cov) == len(MODULE_TYPES) and all(v > COVERAGE_MIN_DIFF for v in cov.values())
        res["passed"] = res["passed"] and res["coverage_ok"]
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
    print(json.dumps({k: v for k, v in res.items() if k not in ("git_commit", "created_at")}, indent=2))
    if not res["passed"]:
        raise SystemExit("LoRA serving check failed")


if __name__ == "__main__":
    from calign.inference.process import run_and_exit

    run_and_exit(main)  # vLLM + LoRA processes do not exit on their own (see calign.inference.process)
