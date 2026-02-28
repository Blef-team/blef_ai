"""
JSON control-plane watcher for NFSP training.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from typing import Any, Dict, Optional, Tuple, Callable

CONTROL_SCHEMA: Dict[str, Dict[str, Any]] = {
    "eta": {"type": "float", "bounds": (0.0, 1.0), "setter": "set_eta", "current": lambda agent, env: float(agent.cfg.anticipatory_eta)},
    "epsilon": {"type": "float", "bounds": (0.0, 1.0), "setter": "set_epsilon", "current": lambda agent, env: float(agent._epsilon())},
    "lr_q": {"type": "float", "bounds": (1e-8, 1e-2), "setter": "set_lr_q", "current": lambda agent, env: float(agent.opt_q.param_groups[0]["lr"])},
    "lr_pi": {"type": "float", "bounds": (1e-8, 1e-2), "setter": "set_lr_pi", "current": lambda agent, env: float(agent.opt_pi.param_groups[0]["lr"])},
    "train_rl_every": {"type": "int", "bounds": (1, 512), "setter": "set_train_rl_every", "current": lambda agent, env: int(agent.cfg.train_rl_every)},
    "batch_rl": {"type": "int", "bounds": (16, 4096), "setter": "set_batch_rl", "current": lambda agent, env: int(agent.cfg.batch_rl)},
    "train_sl_every": {"type": "int", "bounds": (1, 512), "setter": "set_train_sl_every", "current": lambda agent, env: int(agent.cfg.train_sl_every)},
    "batch_sl": {"type": "int", "bounds": (32, 8192), "setter": "set_batch_sl", "current": lambda agent, env: int(agent.cfg.batch_sl)},
    "tau": {"type": "float", "bounds": (0.0, 0.5), "setter": "set_target_tau", "current": lambda agent, env: float(agent.cfg.target_tau)},
    "hard_target_interval": {"type": "int", "bounds": (0, 100_000), "setter": "set_hard_target_interval", "current": lambda agent, env: int(agent.cfg.hard_target_interval)},
    "n_step": {"type": "int", "bounds": (1, 32), "setter": "set_n_step", "current": lambda agent, env: int(getattr(agent, "_active_n_step", agent.cfg.n_step))},
    "max_cards": {"type": "int", "bounds": (1, 11), "setter": "set_max_cards", "current": lambda agent, env: int(getattr(env, "max_cards", 1))},
    "burst_rl_updates_on_reward": {"type": "int", "bounds": (0, 16), "setter": "set_burst_rl_updates", "current": lambda agent, env: int(agent.cfg.burst_rl_updates_on_reward)},
    "burst_reward_threshold": {"type": "float", "bounds": (-1.0, 1.0), "setter": "set_burst_reward_threshold", "current": lambda agent, env: float(agent.cfg.burst_reward_threshold)},
    "check_prob": {"type": "float", "bounds": (0.0, 1.0), "setter": "set_check_explore_prob", "current": lambda agent, env: float(getattr(agent, "_check_explore_prob", 0.0))},
    "sl_learning_off": {"type": "bool", "setter": "set_sl_learning_off", "current": lambda agent, env: bool(agent.cfg.sl_learning_off)},
    "n_agents": {"type": "int", "bounds": (2, 8), "setter": "set_n_agents", "current": lambda agent, env: int(getattr(env, "n_agents", 2))},
    "min_round": {"type": "int", "bounds": (1, 20), "setter": "set_min_round", "current": lambda agent, env: int(getattr(agent.cfg, "min_round", 1))}
}


def _coerce_value(entry: Dict[str, Any], value: Any) -> Any:
    if entry["type"] == "float":
        if isinstance(value, (int, float)):
            coerced = float(value)
        else:
            raise TypeError("expected number")
    elif entry["type"] == "int":
        if isinstance(value, (int, float)) and float(value).is_integer():
            coerced = int(value)
        else:
            raise TypeError("expected integer")
    elif entry["type"] == "bool":
        if isinstance(value, bool):
            coerced = bool(value)
        else:
            raise TypeError("expected boolean")
    else:
        raise TypeError("unsupported type")

    low, high = entry.get("bounds", (None, None))
    if low is not None and coerced < low:
        raise ValueError(f"value {coerced} below {low}")
    if high is not None and coerced > high:
        raise ValueError(f"value {coerced} above {high}")
    return coerced


def write_control_file(path: str, payload: Dict[str, Any]) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".tmp_control_plane_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def build_control_snapshot(agent, env, cooldown_steps: int = 50_000) -> Dict[str, Any]:
    """Return a bootstrap control-plane payload with null overrides."""
    overrides = {key: None for key in CONTROL_SCHEMA}
    defaults = {}
    for key, entry in CONTROL_SCHEMA.items():
        getter: Callable = entry.get("current")
        try:
            defaults[key] = getter(agent, env)
        except Exception:
            defaults[key] = None
    snapshot = {
        "meta": {
            "cooldown_steps": int(max(0, cooldown_steps)),
        },
        "defaults": defaults,
        "overrides": overrides,
    }
    return snapshot


class JsonControlPlane:
    """
    Watches a JSON file and applies overrides to an NFSPAgent + env.
    """

    def __init__(self, path: str, *, cooldown_steps: int = 50_000, verbose: bool = True, schema: Optional[Dict[str, Dict[str, Any]]] = None):
        self.path = path
        self.cooldown_steps = int(max(0, cooldown_steps))
        self.verbose = verbose
        self.schema = schema or CONTROL_SCHEMA
        self._last_hash: Optional[str] = None
        self._last_error_hash: Optional[str] = None
        self._last_pending_hash: Optional[str] = None
        self._last_applied_step: int = -10**12

    def tick(self, agent, env, metrics: Dict[str, Any]) -> None:
        raw = self._read_file()
        if raw is None:
            return
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        if digest == self._last_hash:
            return
        try:
            meta, overrides = self._parse_payload(raw)
            normalized = self._validate_overrides(overrides)
        except Exception as exc:
            if digest != self._last_error_hash and self.verbose:
                print(f"[control] invalid control-plane file: {exc}")
            self._last_error_hash = digest
            return

        if digest != self._last_error_hash:
            self._last_error_hash = None

        if meta:
            self._maybe_update_cooldown(meta)

        step = int(metrics.get("step", getattr(agent, "total_env_steps", 0)))
        remaining = self.cooldown_steps - (step - self._last_applied_step)
        if remaining > 0:
            if digest != self._last_pending_hash and self.verbose:
                print(f"[control] cooldown active ({remaining} steps remaining); overrides queued")
            self._last_pending_hash = digest
            return

        changes = self._apply_overrides(agent, env, normalized, step)
        self._last_applied_step = step
        self._last_hash = digest
        self._last_pending_hash = None
        if not changes:
            return
        if self.verbose:
            for change in changes:
                print(f"[control] step {step}: {change}")

    # --- internals ---
    def _read_file(self) -> Optional[str]:
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                return handle.read()
        except FileNotFoundError:
            return None

    def _parse_payload(self, raw: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("control-plane JSON must be an object")
        meta = data.get("meta", {})
        overrides = data.get("overrides")
        if overrides is None:
            # allow root-level overrides when no explicit section provided
            overrides = {k: v for k, v in data.items() if k not in {"meta", "defaults"}}
        if not isinstance(overrides, dict):
            raise ValueError("overrides must be an object")
        return meta if isinstance(meta, dict) else {}, overrides

    def _maybe_update_cooldown(self, meta: Dict[str, Any]) -> None:
        if "cooldown_steps" in meta:
            try:
                value = int(meta["cooldown_steps"])
                value = max(0, value)
            except Exception:
                if self.verbose:
                    print("[control] ignoring invalid cooldown_steps in meta")
                return
            if value != self.cooldown_steps and self.verbose:
                print(f"[control] cooldown updated: {self.cooldown_steps} -> {value}")
            self.cooldown_steps = value

    def _validate_overrides(self, overrides: Dict[str, Any]) -> Dict[str, Optional[Any]]:
        normalized: Dict[str, Optional[Any]] = {}
        for key, raw_value in overrides.items():
            if key not in self.schema:
                raise ValueError(f"unknown override key '{key}'")
            if raw_value is None:
                normalized[key] = None
                continue
            entry = self.schema[key]
            coerced = _coerce_value(entry, raw_value)
            normalized[key] = coerced
        return normalized

    def _apply_overrides(self, agent, env, overrides: Dict[str, Optional[Any]], step: int) -> list[str]:
        changes: list[str] = []
        for key, value in overrides.items():
            entry = self.schema.get(key)
            if not entry:
                continue
            getter: Callable = entry.get("current", lambda *_: None)
            before = None
            try:
                before = getter(agent, env)
            except Exception:
                pass
            setter_name = entry["setter"]
            setter = getattr(agent, setter_name)
            try:
                if value is None:
                    if key in ["max_cards", "n_agents"]:
                        result = setter(None, env=env, pin=False)
                    else:
                        try:
                            result = setter(None, pin=False)
                        except TypeError:
                            result = setter(None)
                else:
                    if key in ["max_cards", "n_agents"]:
                        result = setter(value, env=env)
                    else:
                        result = setter(value)
            except Exception as exc:
                if self.verbose:
                    print(f"[control] failed to apply {key}: {exc}")
                continue
            after = None
            try:
                after = getter(agent, env)
            except Exception:
                after = result
                if value is None:
                    if before is not None:
                        changes.append(f"{key}: unpinned (was {before})")
                    else:
                        changes.append(f"{key}: unpinned")
            else:
                if value is None:
                    if before is not None:
                        changes.append(f"{key}: unpinned (was {before})")
                else:
                    if before == after:
                        changes.append(f"{key}: pinned at {after}")
                    else:
                        changes.append(f"{key}: {before} -> {after}")
        if changes:
            if not hasattr(agent, "_control_plane_events"):
                agent._control_plane_events = []
            try:
                agent._control_plane_events.append({"step": step, "changes": list(changes)})
            except Exception:
                pass
        return changes


__all__ = ["JsonControlPlane", "CONTROL_SCHEMA", "build_control_snapshot", "write_control_file"]
