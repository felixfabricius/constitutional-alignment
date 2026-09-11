"""Which Gemma Scope 2 SAE features point along each probe direction (cosine with decoder rows), with Neuronpedia labels.

CLI (CPU, local; downloads one params.safetensors per layer, ~2.6 GB for width 65k):
    uv run python -m calign.probe.sae --probes outputs/probes/<run> [--config configs/probe.yaml] [--out outputs/probe_sae/<run>]
        [--layers 16,31] [--no-neuronpedia] [--dry-run] [--limit N]

SAEs: `google/gemma-scope-2-27b-it`, folder `resid_post/layer_{L}_width_{width}_l0_{l0}` (widths 16k/65k/262k/1m,
l0 small/medium/big), file `params.safetensors`; the decoder matrix (n_features, d_model) is found by key name
(W_dec / w_dec / decoder.weight) or, failing that, by shape. Rows are unit-normalised before the cosine.
Caveat (plan 3.8): the SAEs were trained on the base IT model while the probes come from the SFT model; the top
cosines are reported next to the random baseline 1/sqrt(d) and should be read as suggestive.

Labels: Neuronpedia `GET /api/feature/{model}/{layer}-gemmascope-2-res-{width}/{index}` (no key needed); the
`explanations` list (description + model) is stored when present, otherwise the top activating tokens (`pos_str`).
Responses are cached under data/cache/neuronpedia/.

Output: features.jsonl (probe_id, layer, feature, cosine, rank, sign, explanation, top_tokens) and summary.{json,md}.
"""

from __future__ import annotations

import argparse
import json
import logging
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from calign.config import add_common_args, effective_limit, git_commit, new_run_dir, sha256_file
from calign.paths import CACHE_DIR, REPO_ROOT, hf_token
from calign.probe.config import ProbeConfig, load_probe_config
from calign.schemas import ProbeRecord, read_jsonl, utc_now_iso, write_json

LOGGER = logging.getLogger(__name__)
RUN_KIND = "probe_sae"
DECODER_KEYS = ("W_dec", "w_dec", "decoder.weight", "W_D")


def sae_folder(layer: int, width: str, l0: str) -> str:
    return f"resid_post/layer_{layer}_width_{width}_l0_{l0}"


def neuronpedia_source(layer: int, width: str) -> str:
    return f"{layer}-gemmascope-2-res-{width}"


def find_decoder_key(keys: dict[str, tuple[int, ...]], d_model: int) -> str:
    """Decoder tensor name: a known key, else the unique 2-D tensor shaped (n_features, d_model) with n > d."""
    for k in DECODER_KEYS:
        if k in keys:
            return k
    cands = [k for k, shape in keys.items() if len(shape) == 2 and shape[1] == d_model and shape[0] > d_model]
    if len(cands) != 1:
        raise KeyError(f"cannot identify the decoder among {keys}")
    return cands[0]


def load_decoder(path: Path, d_model: int) -> np.ndarray:
    """Unit-normalised decoder rows (n_features, d_model) float32 from a Gemma Scope params.safetensors."""
    from safetensors import safe_open

    with safe_open(str(path), framework="np") as f:
        shapes = {k: tuple(f.get_slice(k).get_shape()) for k in f.keys()}
        key = find_decoder_key(shapes, d_model)
        w = np.asarray(f.get_tensor(key), dtype=np.float32)
    norms = np.linalg.norm(w, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return w / norms


def download_params(repo_id: str, layer: int, width: str, l0: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id, f"{sae_folder(layer, width, l0)}/params.safetensors", token=hf_token()))


def top_features(decoder: np.ndarray, direction: np.ndarray, k: int) -> list[dict]:
    """Top-k features by cosine in each sign: rank 1 = largest positive / most negative cosine."""
    cos = decoder @ (direction / np.linalg.norm(direction))
    out = []
    for sign, order in ((1, np.argsort(-cos)), (-1, np.argsort(cos))):
        for rank, idx in enumerate(order[:k], 1):
            out.append({"feature": int(idx), "cosine": float(cos[idx]), "rank": rank, "sign": sign})
    return out


def cosine_stats(decoder: np.ndarray, direction: np.ndarray) -> dict:
    cos = decoder @ (direction / np.linalg.norm(direction))
    return {
        "n_features": int(len(cos)),
        "max_abs_cos": float(np.abs(cos).max()),
        "p99_abs_cos": float(np.percentile(np.abs(cos), 99)),
        "random_baseline_abs_cos": float(1 / np.sqrt(decoder.shape[1])),
        "n_above_5x_baseline": int((np.abs(cos) > 5 / np.sqrt(decoder.shape[1])).sum()),
    }


# ---------------------------------------------------------------------------- Neuronpedia
def parse_feature_json(d: dict) -> dict:
    exps = d.get("explanations") or []
    best = exps[0] if exps else {}
    return {
        "explanation": best.get("description"),
        "explanation_model": best.get("explanationModelName") or best.get("explanationModel"),
        "n_explanations": len(exps),
        "top_tokens": [str(t) for t in (d.get("pos_str") or [])[:10]],
    }


def fetch_feature(model: str, source: str, index: int, cache_dir: Path = CACHE_DIR / "neuronpedia") -> dict | None:
    cache = cache_dir / model / source / f"{index}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    url = f"https://www.neuronpedia.org/api/feature/{model}/{source}/{index}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "calign/0.1"}), timeout=30) as r:
            data = json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as e:
        LOGGER.warning("neuronpedia %s/%s/%d: %s", model, source, index, e)
        return None
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data), encoding="utf-8")
    return data


# ---------------------------------------------------------------------------- main
def select_probes(probes: list[ProbeRecord], best_per_spec: bool, probe_ids: list[str]) -> list[ProbeRecord]:
    """The best probe (val AUROC) per label spec and/or explicitly named probes, in run order."""
    keep = set(probe_ids)
    unknown = keep - {p.probe_id for p in probes}
    if unknown:
        raise SystemExit(f"unknown probe ids: {sorted(unknown)}")
    if best_per_spec:
        for spec in {p.label_spec for p in probes}:
            cands = [p for p in probes if p.label_spec == spec and p.val_metrics.get("auroc") is not None]
            if cands:
                keep.add(max(cands, key=lambda p: p.val_metrics["auroc"]).probe_id)
    return [p for p in probes if p.probe_id in keep]


def run(
    probes: list[ProbeRecord],
    dirs: np.ndarray,
    cfg: ProbeConfig,
    layers: list[int],
    use_neuronpedia: bool,
    neuronpedia_top: int,
    decoder_loader: Any = None,
    exclude_dims: list[int] | None = None,
) -> tuple[list[dict], dict]:
    """`exclude_dims`: residual coordinates zeroed in both the probe direction and every decoder row before the cosine
    (e.g. Gemma 3's massive-activation dims 104 and 2733, which dominate many difference-of-means directions and make
    raw cosines rank features by their weight on those two coordinates)."""
    decoder_loader = decoder_loader or (
        lambda L: load_decoder(download_params(cfg.sae.repo_id, L, cfg.sae.width, cfg.sae.l0), dirs.shape[1])
    )
    ex = list(exclude_dims or [])

    def strip(x: np.ndarray) -> np.ndarray:
        if not ex:
            return x
        x = x.copy()
        x[..., ex] = 0.0
        n = np.linalg.norm(x, axis=-1, keepdims=True)
        n[n == 0] = 1.0
        return x / n

    rows: list[dict] = []
    stats: dict[str, dict] = {}
    for L in layers:
        ps = [p for p in probes if p.layer == L]
        if not ps:
            continue
        LOGGER.info("layer %d: loading decoder (%s, l0 %s)", L, cfg.sae.width, cfg.sae.l0)
        dec = strip(decoder_loader(L))
        src = neuronpedia_source(L, cfg.sae.width)
        for p in ps:
            raw = dirs[p.direction_row]
            d = strip(raw)
            stats[p.probe_id] = cosine_stats(dec, d)
            if ex:
                stats[p.probe_id]["excluded_dims"] = ex
                stats[p.probe_id]["direction_norm_share_excluded"] = float((raw[ex] ** 2).sum() / (raw**2).sum())
            for feat in top_features(dec, d, cfg.sae.top_k):
                row = {"probe_id": p.probe_id, "layer": L, "source": src, **feat}
                if use_neuronpedia and feat["rank"] <= neuronpedia_top:
                    data = fetch_feature(cfg.sae.neuronpedia_model, src, feat["feature"])
                    row.update(parse_feature_json(data) if data else {"explanation": None, "top_tokens": []})
                rows.append(row)
    return rows, stats


def render_markdown(rows: list[dict], stats: dict, cfg: ProbeConfig) -> str:
    L = ["# SAE features along the probe directions (Gemma Scope 2, suggestive)", ""]
    L.append(
        f"SAEs: {cfg.sae.repo_id} resid_post width {cfg.sae.width} l0 {cfg.sae.l0}; labels: Neuronpedia {cfg.sae.neuronpedia_model}"
    )
    for pid, st in stats.items():
        L += ["", f"## {pid}", ""]
        L.append(
            f"max |cos| {st['max_abs_cos']:.3f}, p99 {st['p99_abs_cos']:.3f}, random baseline {st['random_baseline_abs_cos']:.4f}, "
            f"features above 5x baseline: {st['n_above_5x_baseline']} of {st['n_features']}"
        )
        for sign, label in ((1, "positive (constitutional) direction"), (-1, "negative direction")):
            L += ["", f"**{label}**", ""]
            for r in [x for x in rows if x["probe_id"] == pid and x["sign"] == sign][:10]:
                desc = r.get("explanation") or (
                    "tokens: " + " ".join(r.get("top_tokens") or []) if r.get("top_tokens") else ""
                )
                L.append(f"- #{r['feature']} cos {r['cosine']:+.3f}  {desc}")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, default_config=REPO_ROOT / "configs" / "probe.yaml")
    ap.add_argument("--probes", type=Path, required=True)
    ap.add_argument("--layers", default=None, help="comma-separated subset of the probe layers")
    ap.add_argument("--no-neuronpedia", action="store_true")
    ap.add_argument("--neuronpedia-top", type=int, default=10, help="fetch labels for the top-N features per sign")
    ap.add_argument("--best-per-spec", action="store_true", help="only the best probe (val AUROC) of each label spec")
    ap.add_argument(
        "--exclude-dims", default=None, help="comma-separated residual dims zeroed before the cosine (e.g. 104,2733)"
    )
    ap.add_argument("--probe-ids", default=None, help="comma-separated probe ids to include (added to --best-per-spec)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_probe_config(args.config)
    from safetensors.numpy import load_file

    probes = read_jsonl(args.probes / "probes.jsonl", ProbeRecord)
    dirs = load_file(str(args.probes / "directions.safetensors"))["directions"]
    if args.best_per_spec or args.probe_ids:
        probes = select_probes(probes, args.best_per_spec, args.probe_ids.split(",") if args.probe_ids else [])
    layers = [int(x) for x in args.layers.split(",")] if args.layers else sorted({p.layer for p in probes})
    limit = effective_limit(args)
    if limit:
        probes = probes[:limit]
        layers = [L for L in layers if any(p.layer == L for p in probes)]
    run_dir = new_run_dir(
        RUN_KIND, {"probe": cfg.model_dump(), "probes_run": str(args.probes)}, out=args.out, dry_run=args.dry_run
    )
    exclude = [int(x) for x in args.exclude_dims.split(",")] if args.exclude_dims else None
    rows, stats = run(probes, dirs, cfg, layers, not args.no_neuronpedia, args.neuronpedia_top, exclude_dims=exclude)
    with (run_dir / "features.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summary = {
        "kind": RUN_KIND,
        "stats": stats,
        "sae": cfg.sae.model_dump(),
        "exclude_dims": exclude,
        "provenance": {
            "probes_run": str(args.probes),
            "probes_sha256": sha256_file(args.probes / "probes.jsonl"),
            "git_commit": git_commit(),
            "generated_at": utc_now_iso(),
        },
    }
    write_json(run_dir / "summary.json", summary)
    (run_dir / "summary.md").write_text(render_markdown(rows, stats, cfg), encoding="utf-8")
    print((run_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
