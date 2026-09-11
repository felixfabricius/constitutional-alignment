import json

import yaml

from calign.schemas import Condition, GenerationRecord, Message, ModelRef, Sampling, read_jsonl, write_jsonl
from calign.train import push_to_hub
from calign.validate import judge as judge_mod


def rec(sid: str) -> GenerationRecord:
    return GenerationRecord(
        scenario_id=sid,
        source="moralchoice_high",
        split="probe_train",
        model=ModelRef(name="m", path="p", stage="base"),
        condition=Condition(constitution_in_prompt=True, prompt_variant="full"),
        sampling=Sampling(temperature=0.7, max_tokens=512),
        messages=[Message(role="user", content="u")],
        prompt_text="p",
        response_text="r",
    )


def test_judge_limit_keeps_all_records(tmp_path, monkeypatch):
    records = [rec(f"H_{i}") for i in range(5)]
    write_jsonl(tmp_path / "records.jsonl", records)

    class FakeClient:
        def __init__(self, **_):
            pass

        def dump_usage(self, path):
            return {"total_cost_usd": 0.0}

    async def fake_judge(recs, cfg, client, use_batches):
        return [r.model_copy(update={"extra": {**r.extra, "judged": True}}) for r in recs]

    monkeypatch.setattr(judge_mod, "ClaudeClient", FakeClient)
    monkeypatch.setattr(judge_mod, "judge_records", fake_judge)
    judge_mod.main(["--run-dir", str(tmp_path), "--limit", "2"])
    out = read_jsonl(tmp_path / "records.jsonl", GenerationRecord)
    assert [r.scenario_id for r in out] == [f"H_{i}" for i in range(5)]  # nothing dropped
    assert [bool(r.extra.get("judged")) for r in out] == [True, True, False, False, False]


def _fake_sft_run(tmp_path):
    run = tmp_path / "sft_pilot"
    (run / "merged").mkdir(parents=True)
    (run / "adapter").mkdir()
    (run / "merged" / "config.json").write_text("{}", encoding="utf-8")
    (run / "adapter" / "adapter_config.json").write_text("{}", encoding="utf-8")
    cfg = {
        "base_model": "google/gemma-3-27b-it",
        "train_file": "data/sft/train.jsonl",
        "max_seq_len": 2048,
        "lora": {"r": 64, "alpha": 64, "dropout": 0.05, "target_modules": "regex"},
        "train": {"epochs": 2, "learning_rate": 1e-4, "per_device_batch_size": 1, "gradient_accumulation_steps": 32},
    }
    (run / "resolved_config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    (run / "run_meta.json").write_text(json.dumps({"git_commit": "abc123", "git_dirty": False}), encoding="utf-8")
    (run / "peft_summary.json").write_text(json.dumps({"n_lora_modules": 434, "gpu_peak_reserved_gb": 70.1}))
    (run / "data_stats.json").write_text(json.dumps({"train_file_sha256": "f00d", "train": {"n_examples": 651}}))
    (run / "merged" / "merge_manifest.json").write_text(json.dumps({"kl_per_prompt": [1e-4]}), encoding="utf-8")
    return run


def test_model_card_has_provenance(tmp_path):
    card = push_to_hub.build_model_card(_fake_sft_run(tmp_path), "felixfabricius/x")
    assert card.startswith("---\nbase_model: google/gemma-3-27b-it\n")
    for needle in (
        "r=64",
        "effective batch 32",
        "LoRA modules: 434",
        "sha256 f00d",
        "abc123",
        "[0.0001]",
        "Gemma Terms",
    ):
        assert needle in card, needle


def test_push_dry_run_makes_no_network_calls(tmp_path, capsys, monkeypatch):
    import huggingface_hub

    def boom(*a, **k):
        raise AssertionError("network call in dry run")

    monkeypatch.setattr(huggingface_hub, "HfApi", boom)
    run = _fake_sft_run(tmp_path)
    push_to_hub.main(["--run-dir", str(run), "--repo", "felixfabricius/x", "--dry-run"])
    out = capsys.readouterr().out
    assert "adapter_config.json" in out and "merge_manifest.json" in out
    assert not (run / "push_manifest.json").exists()
