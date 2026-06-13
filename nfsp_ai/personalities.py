"""Play-style "personality" layer for the Blef NFSP production agent.

A personality is *data*, not a new model: one shared set of NFSP weights is
expressed as many distinct opponents by (1) choosing the policy *source*, then
(2) applying a small, bounded *overlay* to that source's action logits, then
(3) modulating the overlay with a stateless per-move *mood*. Morana (CFR) is a
separate deployment and is intentionally absent here.

Why this shape (and what it is NOT):
  The overlay alone — biasing toward high/low-probability claims — is just a
  parametrised conservative heuristic (cf. ``conservative_ai``). To get genuine
  *NFSP-based* character the model is the PRIMARY axis and the overlay is a
  bounded nudge on top, scaled to the policy's own logit spread so a strong
  god stays NFSP-dominated and only a loud/quirky god lets the overlay show.

Axes
----
source     "pi"  average policy  (balanced Nash approximation; prod default)
           "q"   best-response head (sharper, greedier, more exploitable)
           "conservative" / "conservative_crawling"  existing heuristic engines
                 (delegated wholesale — a deliberately simple, different feel)
tau_model  temperature on the chosen head's OWN logits (chaos knob); <1 sharpens
risk       honesty<->bluff on the PRIVATE prob p_priv(a)=P(set exists | my hand)
           risk>0 bluffy (claims unlikely-true sets), risk<0 honest
guard      concealment<->readability on the divergence d(a)=p_priv(a)-pub(a)
           guard>0 unreadable (avoids hand-revealing bets), guard<0 leaky/naive
susp       check threshold on p_priv(last): susp>0 paranoid, susp<0 trusting
tempo      betting cadence (index distance only, NOT a truth proxy):
           tempo<0 minimal legal raise, tempo>0 big senior leaps
mood       stateless modulation of the above, derived from the per-move state
           (card standing, game depth, round parity, last-bet plausibility)

Honest limits: fixed policies cannot model a specific opponent across games, so
"reader"/"manipulator" gods are realised as probability-grounded heuristics +
phase shifts, not online learning. Cross-round memory (tilt after a specific
loss, multi-round traps) is unavailable; standing/depth proxies stand in.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from shared.game_utils import GameRules  # noqa: F401  (kept for callers/tests)
from shared.probabilities.dynamic_probabilities import get_bet_probabilities
from nfsp_ai.nfsp_run_local import _last_bet_action_id

# --------------------------------------------------------------------------- #
# Overlay gains. The overlay is added in LOG-PROBABILITY units on top of the
# chosen head's (temperature-scaled) policy: a bias of +b multiplies an
# action's probability by e^b before re-normalisation. This is head-agnostic
# (pi vs q) and naturally bounded, so a confident NFSP move is nudged, not
# overridden. Gains tuned against tools/personality_probe.py so that strong
# gods stay NFSP-dominated and only loud/quirky gods visibly deviate.
# --------------------------------------------------------------------------- #
BETA = 0.40          # global overlay gain (log-prob units)
TEMPO_W = 0.7        # cadence weight inside the bet bias
CHECK_W = 1.5        # suspicion weight on the check action
SUSP_THETA = 0.5     # neutral check threshold on p_priv(last)
Z_CLIP = 2.5         # clip on standardized probability features
BIAS_CLIP = 3.0      # clip on the total per-action bias (log-prob units)
# Chaos = in-distribution top-k sampling over the SCULPTED policy (never
# uniform/random): keep the k best actions and mix among them with a mild
# in-set temperature. Temperature on the raw head was discarded — the NFSP
# policy is so peaked that any tau>~1.2 collapses it toward uniform and the
# bot just plays random senior bluffs (see tools/personality_probe.py diag).
KC = 2.0             # chaos -> extra top-k actions:  k = round(1 + KC*chaos)
TIN = 0.7            # chaos -> in-set temperature:   tau = 1 + TIN*chaos


@dataclass(frozen=True)
class PersonalityConfig:
    source: str = "pi"
    risk: float = 0.0
    guard: float = 0.0
    susp: float = 0.0
    tempo: float = 0.0
    chaos: float = 0.0
    mood: str = "none"
    mood_params: dict = field(default_factory=dict)
    note: str = ""  # design rationale (also usable as flavor/debug text)
    # Inference-only obs override (no retraining). For each non-CHECK
    # action_id in game.history whose actor matches the selector, set BOTH
    # private_priors[id] and public_priors[id] to 1.0. Logical consistency
    # (public=1 ↔ private=1) means we override both slices simultaneously.
    # Values: any subset of {"self", "opponent"}. Empty tuple = no-op.
    trust_history: tuple = ()


@dataclass
class EffectiveCfg:
    source: str
    risk: float
    guard: float
    susp: float
    tempo: float
    chaos: float
    tau_model: float
    sample: bool


_DELEGATED_SOURCES = {"conservative", "conservative_crawling", "cfr"}


# --------------------------------------------------------------------------- #
# The pantheon (Morana = CFR, deployed separately, intentionally not here).
# Knob scale is roughly [-2, +2]; difficulty falls out of distance from the
# baseline's near-equilibrium balance (bigger distortion ⇒ more exploitable).
# --------------------------------------------------------------------------- #
PERSONALITIES: dict[str, PersonalityConfig] = {
    # --- The difficulty wall: SOTA, lightly tinted -------------------------- #
    "perun": PersonalityConfig(
        source="q", risk=0.15, guard=-0.15, susp=-0.1, tempo=0.2, chaos=0.0,
        mood="standing_push", mood_params={"k": 0.35},
        note="Warlord. Best-response (Q) head is itself the decisive/greedy "
             "voice; only a light overlay on top (the Q head is overlay-"
             "sensitive). Pushes a little harder when ahead. The wall: "
             "near-NFSP strength, faintly bold.",
    ),
    # --- Aggressors -------------------------------------------------------- #
    "czernobog": PersonalityConfig(
        source="pi", risk=1.5, guard=-0.4, susp=0.3, tempo=0.4, chaos=0.9,
        mood="none",
        note="Terror. The loudest aggressor: high-risk pressure on the "
             "(controllable) pi head, volatile and readable (low guard); can "
             "implode (high risk+chaos) but the overlay is bounded so it stays "
             "a live opponent, not a free win.",
    ),
    "dazhbog": PersonalityConfig(
        source="pi", risk=0.7, guard=0.1, susp=-0.3, tempo=0.2, chaos=0.3,
        mood="ego_behind", mood_params={"k": -1.0},
        note="King. Proud 'noble' bluffs; rarely checks and pride makes him "
             "refuse to challenge even when behind (ego-driven misreads). On "
             "the pi head so the bluff magnitude stays controlled.",
    ),
    # --- Disciplined / strong --------------------------------------------- #
    "svetovid": PersonalityConfig(
        source="pi", risk=-0.4, guard=1.0, susp=1.2, tempo=0.0, chaos=-0.3,
        mood="none",
        note="Oracle. Disciplined and unreadable (high guard, sharp tau); "
             "calls implausible bets hard via the p_priv(last) threshold — "
             "real probability-grounded 'reading', not faked adaptivity.",
    ),
    "mokosh": PersonalityConfig(
        source="pi", risk=-0.4, guard=0.8, susp=0.4, tempo=-2.0, chaos=-0.2,
        mood="tighten_late", mood_params={"kr": 0.8, "ks": 1.2},
        note="Weaver. Measured minimal raises, concealed; tightens and grows "
             "more suspicious as the game deepens (dangerous late).",
    ),
    "triglav": PersonalityConfig(
        source="pi", risk=0.0, guard=0.3, susp=0.0, tempo=0.0, chaos=0.3,
        mood="phase_ramp",
        mood_params={"risk_early": -1.2, "risk_late": 1.5,
                     "tempo_early": -1.0, "tempo_late": 1.5,
                     "guard_early": 0.5, "guard_late": 0.0},
        note="Strategist. Three modes by game depth: conservative early, "
             "analytical mid, aggressive late ('all paths converge').",
    ),
    "veles": PersonalityConfig(
        source="pi", risk=0.0, guard=1.8, susp=0.0, tempo=0.0, chaos=0.0,
        mood="phase_ramp",
        mood_params={"risk_early": -1.0, "risk_late": 1.6,
                     "guard_early": 1.8, "guard_late": 1.8},
        note="Serpent. Maximally unreadable; honest early to build a false "
             "pattern, then bluffs late ('truth is the longest bluff'). "
             "True cross-round traps need memory we lack — phase-shift proxy.",
    ),
    # --- Deceivers / mid ---------------------------------------------------- #
    "rusalka": PersonalityConfig(
        source="pi", risk=0.3, guard=1.0, susp=-0.5, tempo=0.0, chaos=0.2,
        mood="phase_ramp",
        mood_params={"risk_early": 0.2, "risk_late": 1.3,
                     "guard_early": 1.0, "guard_late": 1.0},
        note="Siren. Appears modest and passive (low risk, rarely checks), "
             "tells believable lies (high guard), then strikes late once "
             "opponents have overextended ('come closer').",
    ),
    "mavka": PersonalityConfig(
        source="pi", risk=0.3, guard=1.2, susp=-0.4, tempo=0.0, chaos=0.6,
        mood="none",
        note="Illusion. Ambiguous, contradictory signals (high guard + chaos), "
             "passive checking; hard to read but can drift passive ('did you "
             "imagine me?').",
    ),
    "zorya": PersonalityConfig(
        source="pi", risk=0.15, guard=0.0, susp=0.15, tempo=0.0, chaos=0.2,
        mood="parity_swing", mood_params={"d": 0.8},
        note="Twins. Dawn (even rounds): trusting + aggressive. Dusk (odd): "
             "suspicious + defensive. Alternates each round (round parity), so "
             "the state shift is real and learnable ('dawn promises, dusk "
             "remembers').",
    ),
    "kupala": PersonalityConfig(
        source="pi", risk=0.3, guard=-0.5, susp=0.0, tempo=0.0, chaos=0.4,
        mood="win_ramp", mood_params={"k": 0.9, "kc": 0.6},
        note="Reveler. Momentum player: snowballs (more risk + chaos) while "
             "ahead, tilts down when behind. Standing (card lead) is the "
             "stateless proxy for 'winning' ('dance while the fire burns').",
    ),
    "baba_yaga": PersonalityConfig(
        source="pi", risk=0.6, guard=0.8, susp=0.4, tempo=0.0, chaos=0.5,
        mood="spike_random",
        mood_params={"thr": 0.68, "chaos_spike": 1.2, "tempo_spike": 1.5,
                     "risk_spike": 0.6},
        note="Mind-gamer. Mostly composed, but deterministic per-state spikes "
             "throw sudden 'irrational' high-tempo bluffs to bait reactions "
             "('you came to my forest willingly').",
    ),
    "poludnica": PersonalityConfig(
        source="pi", risk=0.0, guard=0.0, susp=0.0, tempo=0.0, chaos=0.0,
        mood="spike_late",
        mood_params={"thr": 0.55, "risk_spike": 1.8, "chaos_spike": 1.0,
                     "tempo_spike": 1.2, "risk_calm": -0.6, "chaos_calm": -0.2},
        note="Fever. Long passive (honest, decisive) stretches, then abrupt "
             "violent escalation once the game deepens. Predictable spike "
             "timing is the weakness ('the heat breaks weaker minds').",
    ),
    # --- Easy / chaotic / fun ---------------------------------------------- #
    "porevit": PersonalityConfig(
        source="pi", risk=0.9, guard=-1.2, susp=0.0, tempo=-0.4, chaos=1.3,
        mood="chaos_decay", mood_params={"k": 1.2},
        note="Youth. Impulsive, leaky (low guard) and experimental (high "
             "chaos), 'learns' within a game as chaos decays with depth "
             "('why not try?'). Inconsistent ⇒ easy.",
    ),
    "leshy": PersonalityConfig(
        source="pi", risk=0.8, guard=0.0, susp=0.0, tempo=0.0, chaos=0.4,
        mood="random_chaos", mood_params={"base": 0.4, "amp": 1.8},
        note="Trickster. Per-move chaos swings between genius and nonsense; "
             "self-destructive randomness ('lost already?'). Easiest NFSP god.",
    ),
    # --- Caretaker (TRAINED NFSP, replaces the conservative_crawling delegate) - #
    "domovoi": PersonalityConfig(
        source="pi", risk=0.5, guard=-0.5, susp=-0.4, tempo=0.8, chaos=1.8,
        note="House spirit. Sculpt-only silly: high chaos (erratic but "
             "in-distribution moves), readable (negative guard), gullible "
             "(negative susp), overexcited raises (tempo). Trained ckpts "
             "retired 2026-06-13 (runs/.bog_program/retired_domovoi/) — the "
             "honesty-trained caretaker was too strong for the laughing "
             "house-spirit art. 'The house laughs with you.'",
    ),
}

PERSONALITY_NAMES = sorted(PERSONALITIES)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * _clamp(t, 0.0, 1.0)


def _z(arr: np.ndarray) -> np.ndarray:
    """Standardize and clip; zeros when degenerate (constant / singleton)."""
    if arr.size <= 1:
        return np.zeros_like(arr)
    s = float(arr.std())
    if s < 1e-9:
        return np.zeros_like(arr)
    return np.clip((arr - float(arr.mean())) / s, -Z_CLIP, Z_CLIP)


def _state_hash(game_state: dict) -> float:
    """Deterministic per-move pseudo-random value in [0, 1).

    Keyed on the current round's bet history + actor + round so a god's
    "random" behaviour is reproducible for a given decision point (not
    re-rolled on retries) yet varies move to move.
    """
    hist = game_state.get("history", []) or []
    ids = ",".join(str((h or {}).get("action_id", "")) for h in hist)
    key = f"{ids}#{game_state.get('cp_nickname', '')}#{game_state.get('round_number', 0)}"
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) % 100000) / 100000.0


def _mood_signals(game_state: dict) -> dict:
    """Stateless mood signals derived purely from the current game state.

    depth     [0,1] game progression (mean active hand size; everyone starts
              at 1 card and the loser accrues a card each round).
    standing  [-1,1] my position; >0 means I hold FEWER cards than rivals
              (i.e., I am winning) — the only stateless proxy for momentum.
    parity    round_number % 2 (flips most rounds: the loser gains a card).
    """
    players = game_state.get("players", []) or []
    cp = game_state.get("cp_nickname")
    active = [int(p.get("n_cards", 0)) for p in players if int(p.get("n_cards", 0)) > 0]
    avg = (sum(active) / len(active)) if active else 1.0
    my = next((int(p.get("n_cards", 0)) for p in players if p.get("nickname") == cp), 0)
    others = [int(p.get("n_cards", 0)) for p in players
              if p.get("nickname") != cp and int(p.get("n_cards", 0)) > 0]
    avg_others = (sum(others) / len(others)) if others else float(my)
    cap = 6.0
    depth = _clamp((avg - 1.0) / (cap - 1.0), 0.0, 1.0)
    standing = _clamp((avg_others - my) / cap, -1.0, 1.0)
    parity = int(game_state.get("round_number", 1) or 1) % 2
    return {"depth": depth, "standing": standing, "parity": parity}


def derive_effective_cfg(cfg: PersonalityConfig, signals: dict, p_last: float,
                         state_r: float) -> EffectiveCfg:
    """Apply the personality's mood rule to its base knobs for this move."""
    risk, guard, susp, tempo, chaos = cfg.risk, cfg.guard, cfg.susp, cfg.tempo, cfg.chaos
    m, pr = cfg.mood, (cfg.mood_params or {})
    depth, standing, parity = signals["depth"], signals["standing"], signals["parity"]

    if m == "phase_ramp":
        risk = _lerp(pr.get("risk_early", risk), pr.get("risk_late", risk), depth)
        guard = _lerp(pr.get("guard_early", guard), pr.get("guard_late", guard), depth)
        tempo = _lerp(pr.get("tempo_early", tempo), pr.get("tempo_late", tempo), depth)
        chaos = _lerp(pr.get("chaos_early", chaos), pr.get("chaos_late", chaos), depth)
    elif m == "standing_push":
        if standing > 0:
            risk += pr.get("k", 1.0) * standing
            tempo += pr.get("k", 1.0) * standing
    elif m == "win_ramp":
        risk += pr.get("k", 1.0) * standing
        if standing > 0:
            chaos += pr.get("kc", 0.5) * standing
    elif m == "spike_late":
        if depth >= pr.get("thr", 0.55):
            risk = pr.get("risk_spike", 2.0)
            chaos = pr.get("chaos_spike", 1.0)
            tempo = pr.get("tempo_spike", 1.0)
        else:
            risk = pr.get("risk_calm", -0.5)
            chaos = pr.get("chaos_calm", -0.2)
    elif m == "tighten_late":
        risk -= pr.get("kr", 1.0) * depth
        susp += pr.get("ks", 1.0) * depth
    elif m == "chaos_decay":
        chaos -= pr.get("k", 1.5) * depth
    elif m == "ego_behind":
        if standing < 0:
            susp += pr.get("k", -1.5) * abs(standing)
    elif m == "trap_overclaim":
        if p_last is not None and p_last < pr.get("thr", 0.35):
            susp += pr.get("k", 1.5)
    elif m == "parity_swing":
        d = pr.get("d", 1.0)
        if parity == 0:
            risk += d
            susp -= d
        else:
            risk -= d
            susp += d
    elif m == "spike_random":
        if state_r > pr.get("thr", 0.7):
            chaos += pr.get("chaos_spike", 1.0)
            tempo += pr.get("tempo_spike", 1.5)
            risk += pr.get("risk_spike", 0.5)
    elif m == "random_chaos":
        chaos = pr.get("base", 0.5) + pr.get("amp", 1.8) * state_r

    risk = _clamp(risk, -3.0, 3.0)
    guard = _clamp(guard, -3.0, 3.0)
    susp = _clamp(susp, -3.0, 3.0)
    tempo = _clamp(tempo, -3.0, 3.0)
    chaos = _clamp(chaos, -1.0, 2.5)
    return EffectiveCfg(
        source=cfg.source, risk=risk, guard=guard, susp=susp, tempo=tempo,
        chaos=chaos, tau_model=1.0, sample=chaos > 0.05,
    )


def sculpt_logits(logits: torch.Tensor, mask: torch.Tensor, eff: EffectiveCfg, *,
                  pvt, pub, p_last: float, check_id: int) -> torch.Tensor:
    """Add the bounded overlay to the chosen head's masked logits.

    ``logits``/``mask`` are ``[1, A]``. Bets are ids ``[0, check_id)``; the
    check action is at ``check_id``. ``pvt``/``pub`` are length-``check_id``
    probability vectors (private / public) already zeroed below the last bet.
    The chosen head is converted to log-probabilities and the overlay is added
    there (a bias of +b ⇒ ×e^b on that action's probability), clipped, so a
    confident NFSP move is nudged rather than overridden. Illegal actions stay
    ``-inf``.
    """
    m = mask.reshape(-1)
    legal = m > 0
    A = logits.shape[-1]
    neg_inf = torch.finfo(logits.dtype).min

    logp = torch.log_softmax(logits, dim=-1)
    row = logp[0]
    bias = torch.zeros(A, dtype=logp.dtype, device=logp.device)

    legal_idx = [i for i in range(min(check_id, A)) if bool(legal[i].item())]
    if legal_idx:
        pvt_a = np.array([float(pvt[i]) for i in legal_idx], dtype=np.float64)
        pub_a = np.array([float(pub[i]) for i in legal_idx], dtype=np.float64)
        z_pvt = _z(pvt_a)
        z_div = _z(pvt_a - pub_a)
        if len(legal_idx) > 1:
            lo, hi = legal_idx[0], legal_idx[-1]
            rng = max(1, hi - lo)
            rank = (np.array(legal_idx, dtype=np.float64) - lo) / rng
            tempo_term = (rank - 0.5) * 2.0
        else:
            tempo_term = np.zeros(1, dtype=np.float64)
        bet_bias = BETA * (
            -eff.risk * z_pvt - eff.guard * z_div + eff.tempo * TEMPO_W * tempo_term
        )
        bet_bias = np.clip(bet_bias, -BIAS_CLIP, BIAS_CLIP)
        for k, i in enumerate(legal_idx):
            bias[i] = float(bet_bias[k])

    if check_id < A and bool(legal[check_id].item()):
        cb = BETA * eff.susp * CHECK_W * (SUSP_THETA - float(p_last)) * 2.0
        bias[check_id] = float(np.clip(cb, -BIAS_CLIP, BIAS_CLIP))

    out = (row + bias).unsqueeze(0)
    return out.masked_fill(m == 0, neg_inf)


def _select(sculpted: torch.Tensor, chaos: float) -> int:
    """Pick an action from the sculpted policy.

    chaos<=0 → argmax (decisive). chaos>0 → coherent in-distribution sampling:
    keep the top ``k = round(1 + KC*chaos)`` actions of the sculpted policy and
    sample among them with a mild in-set temperature. The bot only ever varies
    among moves it already rates highly — never a uniform-random senior bluff.
    """
    if chaos <= 0.05:
        return int(sculpted.argmax(dim=-1).item())
    probs = torch.softmax(sculpted, dim=-1).reshape(-1)
    nz = int((probs > 0).sum().item())
    if nz <= 1:
        return int(probs.argmax().item())
    k = min(nz, max(2, int(round(1 + KC * chaos))))
    topv, topi = probs.topk(k)
    tau = 1.0 + TIN * chaos
    w = topv.double().clamp_min(1e-12) ** (1.0 / tau)
    total = float(w.sum())
    if total <= 0:
        return int(topi[0].item())
    j = int(torch.multinomial(w / w.sum(), 1).item())
    return int(topi[j].item())


# --------------------------------------------------------------------------- #
# Entry points used by the production agent
# --------------------------------------------------------------------------- #
def resolve_personality(game_state: dict) -> Optional[str]:
    """Resolve the personality id for the current player, or None.

    Order: explicit ``personality`` on the current player, then a top-level
    ``personality`` field, then a prefix match against the actor's nickname
    (e.g. ``"perun_(AI)"`` / ``"veles_2_(AI)"``). The nickname fallback lets the
    image work with only an ``agent_mapping`` env change (no engine code change).
    """
    cp = game_state.get("cp_nickname")
    players = game_state.get("players") or []
    cur = next((p for p in players if p.get("nickname") == cp), None)
    cand = (cur or {}).get("personality") or game_state.get("personality")
    if cand:
        key = str(cand).strip().lower()
        if key in PERSONALITIES:
            return key
    if cp:
        nl = str(cp).strip().lower()
        best = None
        for key in PERSONALITIES:
            if nl.startswith(key) and (best is None or len(key) > len(best)):
                best = key
        if best:
            return best
    return None


def _delegate(source: str, game_state: dict) -> int:
    """Delegate the move to an existing heuristic/CFR engine (lazy import)."""
    if source == "conservative_crawling":
        from conservative_crawling_ai.agent import ConservativeCrawlingAgent
        return int(ConservativeCrawlingAgent.determine_action(game_state))
    if source == "conservative":
        from conservative_ai.agent import ConservativeAgent
        return int(ConservativeAgent.determine_action(game_state))
    if source == "cfr":
        from cfr_ai.agent import determine_action as cfr_determine_action
        return int(cfr_determine_action(game_state))
    raise ValueError(f"Unknown delegated source: {source}")


def is_delegated(name: str) -> bool:
    cfg = PERSONALITIES.get(name)
    return bool(cfg and cfg.source in _DELEGATED_SOURCES)


def personality_action(agent, obs_tensor: torch.Tensor, mask_tensor: torch.Tensor,
                       game_state: dict, spec, pub_prior, name: str) -> int:
    """Compute the action for personality ``name`` over the chosen NFSP head."""
    cfg = PERSONALITIES[name]
    signals = _mood_signals(game_state)
    last_bet_id = _last_bet_action_id(game_state, spec)
    p_last = float(get_bet_probabilities(
        game_state=game_state, for_betting=False, specific_action_id=last_bet_id))
    state_r = _state_hash(game_state)
    eff = derive_effective_cfg(cfg, signals, p_last, state_r)

    head = "q" if eff.source == "q" else "pi"
    logits = agent.policy_logits(obs_tensor, mask_tensor, head=head, tau=1.0)
    pvt = get_bet_probabilities(game_state=game_state, for_betting=True, last_bet=last_bet_id)
    sculpted = sculpt_logits(
        logits, mask_tensor, eff,
        pvt=pvt, pub=pub_prior, p_last=p_last, check_id=int(spec.check_action_id),
    )
    return _select(sculpted, eff.chaos)
