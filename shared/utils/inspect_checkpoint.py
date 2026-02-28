#!/usr/bin/env python3
"""Lightweight inspector for NFSP checkpoint (.pt) files.

Example usage:
    python shared/utils/inspect_checkpoint.py nfsp_blef_20251021013941_7M.pt
"""
from __future__ import annotations

import argparse
import io
import json
import os
import pickle
import sys
import zipfile
from collections import Counter
from typing import Any, Dict, Tuple

import torch
import torch._utils  # type: ignore[attr-defined]

_STORAGE_DTYPE = {
    torch.FloatStorage: "float32",
    torch.DoubleStorage: "float64",
    torch.HalfStorage: "float16",
    torch.BFloat16Storage: "bfloat16",
    torch.LongStorage: "int64",
    torch.IntStorage: "int32",
    torch.ShortStorage: "int16",
    torch.CharStorage: "int8",
    torch.ByteStorage: "uint8",
    torch.BoolStorage: "bool",
}


class FakeStorage:
    __slots__ = ("storage_type", "key", "location", "size", "dtype")

    def __init__(self, storage_type: Any, key: str, location: str, size: int) -> None:
        self.storage_type = storage_type
        self.key = key
        self.location = location
        self.size = size
        self.dtype = _STORAGE_DTYPE.get(
            storage_type, getattr(storage_type, "__name__", str(storage_type))
        )


def _find_zip_root(zf: zipfile.ZipFile) -> str:
    names = [name for name in zf.namelist() if name]
    if not names:
        raise ValueError("Empty checkpoint archive")
    first = names[0]
    return first.split("/", 1)[0]


def _load_metadata(path: str) -> Dict[str, Any]:
    with zipfile.ZipFile(path, "r") as zf:
        root = _find_zip_root(zf)
        data = zf.read(f"{root}/data.pkl")

    storage_meta: Dict[str, Dict[str, Any]] = {}
    tensor_metas = []

    orig_rebuild_tensor = torch._utils._rebuild_tensor_v2
    orig_rebuild_param = torch._utils._rebuild_parameter

    def rebuild_tensor_stub(
        storage: FakeStorage,
        storage_offset: int,
        size: Tuple[int, ...],
        stride: Tuple[int, ...],
        requires_grad: bool,
        backward_hooks: Any,
        metadata: Any = None,
    ) -> Dict[str, Any]:
        tensor_meta = {
            "storage_key": getattr(storage, "key", None),
            "dtype": getattr(storage, "dtype", None),
            "device": getattr(storage, "location", None),
            "storage_size": getattr(storage, "size", None),
            "storage_offset": storage_offset,
            "size": tuple(size),
            "stride": tuple(stride),
            "requires_grad": requires_grad,
        }
        tensor_metas.append(tensor_meta)
        return tensor_meta

    def rebuild_param_stub(
        tensor_meta: Dict[str, Any], requires_grad: bool, backward_hooks: Any
    ) -> Dict[str, Any]:
        tensor_meta = dict(tensor_meta)
        tensor_meta["requires_grad"] = requires_grad
        return tensor_meta

    torch._utils._rebuild_tensor_v2 = rebuild_tensor_stub  # type: ignore[assignment]
    torch._utils._rebuild_parameter = rebuild_param_stub  # type: ignore[assignment]

    try:
        def persistent_load(pid: Any) -> FakeStorage:
            if isinstance(pid, tuple) and pid and pid[0] == "storage":
                _, storage_type, key, location, size, *rest = pid
                storage_meta[key] = {
                    "storage_type": getattr(storage_type, "__name__", str(storage_type)),
                    "location": location,
                    "size": size,
                    "dtype": _STORAGE_DTYPE.get(
                        storage_type, getattr(storage_type, "__name__", str(storage_type))
                    ),
                }
                return FakeStorage(storage_type, key, location, size)
            raise RuntimeError(f"Unexpected persistent id: {pid!r}")

        unpickler = pickle.Unpickler(io.BytesIO(data))
        unpickler.persistent_load = persistent_load
        checkpoint = unpickler.load()
    finally:
        torch._utils._rebuild_tensor_v2 = orig_rebuild_tensor  # type: ignore[assignment]
        torch._utils._rebuild_parameter = orig_rebuild_param  # type: ignore[assignment]

    checkpoint["_tensor_meta"] = tensor_metas
    checkpoint["_storage_meta"] = storage_meta
    return checkpoint


def summarise(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    cfg = checkpoint.get("cfg", {})
    rl_buf = checkpoint.get("rl_buf", {})
    sl_buf = checkpoint.get("sl_buf", {})
    storage_meta = checkpoint.get("_storage_meta", {})
    tensor_meta = checkpoint.get("_tensor_meta", [])

    summary: Dict[str, Any] = {
        "steps": checkpoint.get("steps"),
        "cfg": cfg,
        "rl_buf": rl_buf,
        "sl_buf": sl_buf,
        "num_storages": len(storage_meta),
        "tensor_count": len(tensor_meta),
    }

    control_plane = {
        "pinned_overrides": checkpoint.get("pinned_overrides", {}),
        "pinned_env_overrides": checkpoint.get("pinned_env_overrides", {}),
        "eps_current": checkpoint.get("eps_current"),
        "check_explore_prob": checkpoint.get("check_explore_prob"),
    }
    summary["control_plane"] = control_plane

    if tensor_meta:
        dtype_counts = Counter(meta.get("dtype") for meta in tensor_meta)
        summary["tensor_dtypes"] = dict(dtype_counts)

    if storage_meta:
        sample = {k: storage_meta[k] for k in list(storage_meta.keys())[:5]}
        summary["storage_sample"] = sample

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect an NFSP checkpoint without loading large tensors."
    )
    parser.add_argument("path", help="Path to the .pt checkpoint")
    parser.add_argument(
        "--json", action="store_true", help="Emit the summary as JSON instead of text"
    )
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"error: file not found: {args.path}", file=sys.stderr)
        sys.exit(1)

    try:
        checkpoint = _load_metadata(args.path)
    except Exception as exc:
        print(f"error: failed to inspect {args.path}: {exc}", file=sys.stderr)
        sys.exit(1)

    summary = summarise(checkpoint)

    if args.json:
        json.dump(summary, sys.stdout, indent=2, sort_keys=True)
        print()
        return

    print(f"Checkpoint: {args.path}")
    print(f"  steps: {summary.get('steps')}")
    print("  cfg:")
    for key in sorted(summary.get("cfg", {}).keys()):
        print(f"    {key}: {summary['cfg'][key]}")
    if summary.get("rl_buf"):
        meta = summary["rl_buf"]
        print(
            f"  rl_buf: size={meta.get('size')} capacity={meta.get('capacity')}"
        )
    if summary.get("sl_buf"):
        meta = summary["sl_buf"]
        filled = meta.get("filled", meta.get("size"))
        print(f"  sl_buf: filled={filled} capacity={meta.get('capacity')}")
    if summary.get("control_plane"):
        cp = summary["control_plane"]
        pins = cp.get("pinned_overrides") or {}
        env_pins = cp.get("pinned_env_overrides") or {}
        if pins or env_pins:
            print("  control-plane pins:")
            for key in sorted(pins.keys()):
                print(f"    {key}: {pins[key]}")
            for key in sorted(env_pins.keys()):
                print(f"    env.{key}: {env_pins[key]}")
        eps_cur = cp.get("eps_current")
        chk_prob = cp.get("check_explore_prob")
        if eps_cur is not None or chk_prob is not None:
            print(
                f"  control-plane state: eps_current={eps_cur} check_prob={chk_prob}"
            )
    if "tensor_dtypes" in summary:
        print("  tensor dtypes:")
        for dtype, count in summary["tensor_dtypes"].items():
            print(f"    {dtype}: {count}")
    if "storage_sample" in summary:
        print("  storage sample:")
        for key, meta in summary["storage_sample"].items():
            print(f"    {key}: {meta}")


if __name__ == "__main__":
    main()
