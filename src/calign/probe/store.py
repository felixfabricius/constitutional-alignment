"""Activation storage: safetensors shards of stored positions, plus an index; ActivationRef points into them.

Layout inside a run dir:

    activations/index.json           {"dtype", "d_model", "layers", "positions", "context_variant",
                                      "shards": [{"path": "shard_000.safetensors", "record_ids": [...]}, ...]}
    activations/shard_000.safetensors  tensor "acts": (n_records, n_layers, n_positions, d_model)

`layers` are Gemma Scope layer numbers, `positions` the position names (fixed order). Only the stored positions
are kept; full sequences are never written. `ActivationWriter` streams records into shards; `ActivationStore`
reads them back by record id or as (n, d) matrices for one (layer, position).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from calign.schemas import ActivationRef

ACT_DIR = "activations"
INDEX_FILE = "index.json"
_DTYPES = {"float32": np.float32, "float16": np.float16}


class ActivationWriter:
    def __init__(
        self,
        run_dir: Path,
        layers: list[int],
        positions: list[str],
        d_model: int,
        dtype: str = "float32",
        shard_size: int = 512,
        context_variant: str = "same",
    ) -> None:
        self.dir = Path(run_dir) / ACT_DIR
        self.dir.mkdir(parents=True, exist_ok=True)
        if (self.dir / INDEX_FILE).exists():
            raise FileExistsError(f"{self.dir / INDEX_FILE} exists; activations are written once per run dir")
        self.layers, self.positions, self.d_model = list(layers), list(positions), int(d_model)
        self.dtype, self.shard_size, self.context_variant = dtype, int(shard_size), context_variant
        self._buf: list[np.ndarray] = []
        self._buf_ids: list[str] = []
        self._shards: list[dict] = []
        self._seen: set[str] = set()
        self.refs: dict[str, ActivationRef] = {}

    def add(self, record_id: str, acts: np.ndarray, token_positions: dict[str, int]) -> ActivationRef:
        """`acts` has shape (n_layers, n_positions, d_model); `token_positions` maps position name -> token index."""
        if record_id in self._seen:
            raise ValueError(f"duplicate record id {record_id}")
        expected = (len(self.layers), len(self.positions), self.d_model)
        if tuple(acts.shape) != expected:
            raise ValueError(f"activations for {record_id} have shape {tuple(acts.shape)}, expected {expected}")
        if set(token_positions) != set(self.positions):
            raise ValueError(f"token positions {sorted(token_positions)} != stored positions {self.positions}")
        self._seen.add(record_id)
        self._buf.append(np.asarray(acts, dtype=_DTYPES[self.dtype]))
        self._buf_ids.append(record_id)
        ref = ActivationRef(
            path=f"{ACT_DIR}/{self._shard_name(len(self._shards))}",
            layers=self.layers,
            positions=dict(token_positions),
            row=len(self._buf) - 1,
            context_variant=self.context_variant,
        )
        self.refs[record_id] = ref
        if len(self._buf) >= self.shard_size:
            self._flush()
        return ref

    @staticmethod
    def _shard_name(i: int) -> str:
        return f"shard_{i:03d}.safetensors"

    def _flush(self) -> None:
        if not self._buf:
            return
        from safetensors.numpy import save_file

        name = self._shard_name(len(self._shards))
        save_file({"acts": np.stack(self._buf)}, str(self.dir / name))
        self._shards.append({"path": name, "record_ids": list(self._buf_ids), "n": len(self._buf_ids)})
        self._buf, self._buf_ids = [], []

    def close(self) -> Path:
        self._flush()
        index = {
            "dtype": self.dtype,
            "d_model": self.d_model,
            "layers": self.layers,
            "positions": self.positions,
            "context_variant": self.context_variant,
            "n_records": sum(s["n"] for s in self._shards),
            "shards": self._shards,
        }
        path = self.dir / INDEX_FILE
        path.write_text(json.dumps(index, indent=2), encoding="utf-8")
        return path

    def __enter__(self) -> ActivationWriter:
        return self

    def __exit__(self, *exc) -> None:
        if exc[0] is None:
            self.close()


class ActivationStore:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.dir = self.run_dir / ACT_DIR
        self.index = json.loads((self.dir / INDEX_FILE).read_text(encoding="utf-8"))
        self.layers: list[int] = list(self.index["layers"])
        self.positions: list[str] = list(self.index["positions"])
        self.d_model: int = int(self.index["d_model"])
        self.context_variant: str = self.index.get("context_variant", "same")
        self._loc: dict[str, tuple[int, int]] = {}  # record_id -> (shard idx, row)
        for si, s in enumerate(self.index["shards"]):
            for row, rid in enumerate(s["record_ids"]):
                if rid in self._loc:
                    raise ValueError(f"record id {rid} appears twice in {self.dir / INDEX_FILE}")
                self._loc[rid] = (si, row)
        self._cache: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self._loc)

    def __contains__(self, record_id: str) -> bool:
        return record_id in self._loc

    @property
    def record_ids(self) -> list[str]:
        return list(self._loc)

    def _shard(self, i: int) -> np.ndarray:
        if i not in self._cache:
            from safetensors.numpy import load_file

            self._cache[i] = load_file(str(self.dir / self.index["shards"][i]["path"]))["acts"]
        return self._cache[i]

    def get(self, record_id: str) -> np.ndarray:
        """(n_layers, n_positions, d_model) for one record."""
        si, row = self._loc[record_id]
        return self._shard(si)[row]

    def matrix(self, record_ids: list[str], layer: int, position: str) -> np.ndarray:
        """(n, d_model) float32 rows for `record_ids` at one (layer, position), in the given order."""
        li, pi = self.layers.index(layer), self.positions.index(position)
        out = np.empty((len(record_ids), self.d_model), dtype=np.float32)
        by_shard: dict[int, list[tuple[int, int]]] = {}
        for k, rid in enumerate(record_ids):
            si, row = self._loc[rid]
            by_shard.setdefault(si, []).append((k, row))
        for si, items in by_shard.items():
            acts = self._shard(si)
            ks, rows = zip(*items, strict=True)
            out[list(ks)] = acts[list(rows), li, pi].astype(np.float32)
        return out

    def check_ref(self, record_id: str, ref: ActivationRef) -> None:
        si, row = self._loc[record_id]
        if ref.row != row or ref.path != f"{ACT_DIR}/{self.index['shards'][si]['path']}":
            raise ValueError(f"ActivationRef for {record_id} does not match the index ({ref.path}#{ref.row})")
