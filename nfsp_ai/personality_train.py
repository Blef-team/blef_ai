"""Training-time personality bias toolkit.

Four mechanisms keyed on a single ``PersonalityTrainSpec``:

  A. Terminal-magnitude shaping — multiply the MC terminal credit by a
     per-trajectory scalar derived from the spec, the win/loss outcome, a
     bluff oracle, and a hand-type bias keyed on the actor's bets.

  B. Action-mask restriction — hard-zero forbidden bet ranges and (optionally)
     the CHECK action, with a mandatory safety rail that always leaves at
     least one legal action.

  C. Observation feature zero-mask — zero named obs blocks at vectorize time
     to produce trained-in perceptual deficits. The block layout mirrors
     ``nfsp_ai.nfsp_run_local._compute_obs_dim`` and is resolved from the
     same ``DeckSpec`` fields, so layout drift is impossible.

  E. Opponent-distribution shaping ("honesty bias") — during self-play, bias
     the *opponent's* action selection toward truthful bets (bets whose set
     actually exists, given the opponent's hand + common cards). The
     learning seat's reward, architecture and replay buffer are unchanged;
     only the world the bot trains against is altered. The Nash response to
     an honest-biased opponent population is to *trust* opponent bets — the
     bot ends up Bayesian-updating "bets are usually true," which is exactly
     the trusting-deity trait we want to install. At inference against a
     bluffing population (humans, CFR, equilibrium), the bot is structurally
     exploited on bluffs — the intended failure mode.

     Why this isn't reward shaping (mechanism A): A pushes the value
     function in the same self-play world. The Nash attractor pulls the
     policy back toward equilibrium, and aggressive A-shaping has been
     shown to collapse (see kupala v1, 2026-05-30). E *changes the world*
     itself, so the new optimum genuinely is "trusting" — no fighting
     gradients, no collapse, the trait is the policy's optimal response.

     Why this isn't obs override (proposed mechanism D, found insufficient):
     A Nash-trained policy has redundant truth signals (history,
     last_bet_prob, priors) and is robust to single-channel input distortion.
     We measured ~no behavioural change under inference-time prior override
     (2026-06-07). E shifts the *training distribution* instead, which the
     policy must adapt to.

     Important: E lives ONLY in the training loop. The exported checkpoint
     carries the spec's `opponent_honesty_bias` value for forensics, but
     the inference path ignores it (E is consumed at train time, not at
     serve time).

  (Mechanism D — inference-only obs override — is implemented as a separate
  primitive in ``apply_trust_history_to_obs`` for possible future use, but
  is not bound to any personality after empirical validation showed it does
  not move trained-policy behaviour.)

The spec lives in a JSON file (one personality per file) and is also stamped
into the exported inference checkpoint's ``cfg["personality"]`` so the
production loader can reapply C and B at serve time symmetrically to training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional, Sequence

import numpy as np

from shared.game_utils import (
    GameRules,
    determine_set_existence,
    get_set_details_from_action_id,
)


# Hand-type strings as emitted by shared.game_utils.get_set_details_from_action_id.
KNOWN_HAND_TYPES = (
    "High card",
    "Pair",
    "Two pairs",
    "Straight",
    "Three of a kind",
    "Full house",
    "Flush",
    "Four of a kind",
    "Straight flush",
)

# Obs block names recognised by the blind-spot mask. Order matches the
# layout produced by nfsp_ai.nfsp_run_local.vectorize_obs / _compute_obs_dim.
KNOWN_OBS_BLOCKS = (
    "private_hand",
    "rules_jokers",
    "private_jokers",
    "common_jokers",
    "common_hand",
    "seat_counts",
    "team_flags",       # only present when DeckSpec.team_aware is True
    "round",
    "history",
    "last_bet_prob",
    "private_priors",
    "public_priors",
    "blanks",
)

MAX_PLAYERS = 8  # mirrors nfsp_ai.nfsp_run_local.MAX_PLAYERS


@dataclass(frozen=True)
class PersonalityTrainSpec:
    name: str
    # --- A: terminal-magnitude shaping ---
    win_multiplier: float = 1.0
    loss_multiplier: float = 1.0
    bluff_caught_penalty: float = 0.0       # added to loss scale when actor was caught bluffing
    bluff_call_bonus: float = 0.0           # added to win scale when actor caught a bluff
    hand_type_bias: dict = field(default_factory=dict)  # {set_type: float}; positive = likes
    # --- B: action-mask restriction ---
    forbid_hand_types: tuple = ()
    forbid_check_unless_only: bool = False
    # --- C: observation feature zero-mask ---
    obs_blind_spots: tuple = ()
    # --- E: opponent-distribution shaping family ---
    # Each knob biases the self-play opponent's intended action toward a
    # specific behavioural mode. The bot's learned policy is then the Nash
    # response to that *modified* world, which carries a coherent inverse
    # trait at inference (e.g., honest-biased training -> bot believes
    # claims; aggression-biased training -> bot becomes a patient defender;
    # passivity-biased training -> bot becomes a pressurer).
    #
    # All knobs live in [0.0, 1.0]. 0 = no bias. Higher = stronger
    # distribution shift. Validated mutually exclusive: at most one of the
    # three can be set > 0 per spec (the mechanism conditions on a single
    # opponent style; mixing them muddles the learned trait). Used ONLY at
    # training time; the inference path ignores these fields. All are
    # stamped into the exported ckpt cfg for forensics.

    # Honesty: with prob bias, if opponent's intended bet is a bluff,
    # resample to a truthful alternative (set actually exists in hand).
    # Produces "honest reciprocator" / "trusting" learned policy.
    opponent_honesty_bias: float = 0.0

    # Aggression: with prob bias, replace opponent's intended action with
    # the HIGHEST legal bet (max-escalation). CHECK is never overridden.
    # Produces "patient defender" learned policy — bot trains to wait out
    # escalators.
    opponent_aggression_bias: float = 0.0

    # Passivity: with prob bias, replace opponent's intended action with
    # CHECK if legal, else with the LOWEST legal bet. Produces "pressurer"
    # learned policy — bot trains to exploit perceived weakness.
    opponent_passivity_bias: float = 0.0
    # --- Inference-time variety (NOT used during training) ---
    # greedy: True = argmax over the masked logits (deterministic, "stiff");
    #         False = sample from softmax(masked_logits) (variety, less predictable).
    # head:   "pi" = use the average policy (Nash-shaped); "q" = use the
    #         best-response Q-net (sharper, more aggressive). "pi" + greedy=False
    #         is the canonical "feels human" setting for trained personalities;
    #         "q" + greedy=True is the perun "wall" setting (sculpted only today).
    greedy: bool = True
    head: str = "pi"


def load_spec(path: str) -> PersonalityTrainSpec:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return spec_from_dict(data)


def spec_to_dict(spec: PersonalityTrainSpec) -> dict:
    d = asdict(spec)
    d["forbid_hand_types"] = list(d["forbid_hand_types"])
    d["obs_blind_spots"] = list(d["obs_blind_spots"])
    return d


def spec_from_dict(data: dict) -> PersonalityTrainSpec:
    d = dict(data)
    if "forbid_hand_types" in d:
        d["forbid_hand_types"] = tuple(d["forbid_hand_types"])
    if "obs_blind_spots" in d:
        d["obs_blind_spots"] = tuple(d["obs_blind_spots"])
    for ht in d.get("forbid_hand_types", ()):
        if ht not in KNOWN_HAND_TYPES:
            raise ValueError(
                f"Unknown hand type {ht!r} in forbid_hand_types; known: {KNOWN_HAND_TYPES}"
            )
    for ht in (d.get("hand_type_bias") or {}):
        if ht not in KNOWN_HAND_TYPES:
            raise ValueError(
                f"Unknown hand type {ht!r} in hand_type_bias; known: {KNOWN_HAND_TYPES}"
            )
    for blk in d.get("obs_blind_spots", ()):
        if blk not in KNOWN_OBS_BLOCKS:
            raise ValueError(
                f"Unknown obs block {blk!r} in obs_blind_spots; known: {KNOWN_OBS_BLOCKS}"
            )
    head = d.get("head", "pi")
    if head not in ("pi", "q"):
        raise ValueError(f"head must be 'pi' or 'q', got {head!r}")
    # Mechanism E: at most one opponent-distribution shaping knob may be
    # set above zero per spec. Conditioning on a single opponent style
    # produces a coherent learned trait; mixing them muddles the gradient.
    _e_knobs = {
        "opponent_honesty_bias": float(d.get("opponent_honesty_bias", 0.0) or 0.0),
        "opponent_aggression_bias": float(d.get("opponent_aggression_bias", 0.0) or 0.0),
        "opponent_passivity_bias": float(d.get("opponent_passivity_bias", 0.0) or 0.0),
    }
    active = [k for k, v in _e_knobs.items() if v > 0.0]
    if len(active) > 1:
        raise ValueError(
            f"At most one opponent-distribution shaping bias may be > 0 per spec; "
            f"got: {active}. Pick one direction (honesty / aggression / passivity)."
        )
    for k, v in _e_knobs.items():
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"{k} must be in [0, 1]; got {v}")
    return PersonalityTrainSpec(**d)


# ---------------------------------------------------------------------------
# Layout-block resolver (C)
# ---------------------------------------------------------------------------

def resolve_obs_block_slices(deck_spec) -> dict:
    """Return ``{block_name: (start, end)}`` for every block in the obs vector.

    Mirrors the order in ``nfsp_ai.nfsp_run_local.vectorize_obs`` /
    ``_compute_obs_dim`` exactly. The resolver verifies the computed total
    matches ``deck_spec.obs_dim`` so any drift in the obs layout fails fast.
    """
    card_dim = int(deck_spec.card_feature_dim)
    history_dim = int(deck_spec.history_feature_dim)
    hist_dim = int(deck_spec.hist_dim)
    team_aware = bool(getattr(deck_spec, "team_aware", True))
    slices: dict = {}
    idx = 0

    slices["private_hand"] = (idx, idx + card_dim); idx += card_dim
    slices["rules_jokers"] = (idx, idx + 1); idx += 1
    slices["private_jokers"] = (idx, idx + 1); idx += 1
    slices["common_jokers"] = (idx, idx + 1); idx += 1
    slices["common_hand"] = (idx, idx + card_dim); idx += card_dim
    slices["seat_counts"] = (idx, idx + MAX_PLAYERS); idx += MAX_PLAYERS
    if team_aware:
        slices["team_flags"] = (idx, idx + MAX_PLAYERS); idx += MAX_PLAYERS
    slices["round"] = (idx, idx + 1); idx += 1
    slices["history"] = (idx, idx + history_dim); idx += history_dim
    slices["last_bet_prob"] = (idx, idx + 1); idx += 1
    slices["private_priors"] = (idx, idx + hist_dim); idx += hist_dim
    slices["public_priors"] = (idx, idx + hist_dim); idx += hist_dim
    slices["blanks"] = (idx, idx + 1); idx += 1

    if idx != int(deck_spec.obs_dim):
        raise ValueError(
            f"Layout resolver computed obs_dim={idx} but DeckSpec.obs_dim="
            f"{int(deck_spec.obs_dim)}. Layout out of sync with _compute_obs_dim."
        )
    return slices


def apply_personality_to_obs(
    obs: np.ndarray,
    spec: Optional[PersonalityTrainSpec],
    deck_spec,
) -> np.ndarray:
    """Zero the obs blocks named in ``spec.obs_blind_spots`` (in place)."""
    if spec is None or not spec.obs_blind_spots:
        return obs
    layout = resolve_obs_block_slices(deck_spec)
    for blk in spec.obs_blind_spots:
        rng = layout.get(blk)
        if rng is None:
            # team_flags only exists when DeckSpec.team_aware; ignore silently.
            continue
        start, end = rng
        obs[start:end] = 0.0
    return obs


def apply_trust_history_to_obs(
    obs: np.ndarray,
    deck_spec,
    game: dict,
    trust_history,
) -> np.ndarray:
    """Inference-only obs override (mechanism D). For each non-CHECK
    ``action_id`` in ``game["history"]`` whose actor matches the
    ``trust_history`` selector, set BOTH ``private_priors[id]`` and
    ``public_priors[id]`` to 1.0.

    The dual override exists because the priors are logically linked:
      - public_prior = 0 ⇒ private_prior = 0  (set provably impossible)
      - public_prior = 1 ⇒ private_prior = 1  (set provably exists)
    Overriding only one leaves the obs in an incoherent state.

    ``trust_history`` is a tuple containing any subset of {"self",
    "opponent"}. Empty tuple is a no-op. NEVER applied during training —
    if the policy is trained on distorted priors, it learns to ignore
    them and the trait gets compensated away. This must be a runtime
    intervention on a model that was trained to RELY on the prior
    features.
    """
    if not trust_history:
        return obs
    history = game.get("history") or []
    if not history:
        return obs
    layout = resolve_obs_block_slices(deck_spec)
    pvt_rng = layout.get("private_priors")
    pub_rng = layout.get("public_priors")
    if pvt_rng is None or pub_rng is None:
        return obs
    actor = game.get("cp_nickname")
    check_id = int(deck_spec.check_action_id)
    trust_self = "self" in trust_history
    trust_opp = "opponent" in trust_history
    n = pvt_rng[1] - pvt_rng[0]
    for ev in history:
        try:
            aid = int((ev or {}).get("action_id", -1))
        except Exception:
            continue
        if aid < 0 or aid == check_id or aid >= n:
            continue
        is_self = ((ev or {}).get("player") == actor)
        if (is_self and trust_self) or ((not is_self) and trust_opp):
            obs[pvt_rng[0] + aid] = 1.0
            obs[pub_rng[0] + aid] = 1.0
    return obs


# ---------------------------------------------------------------------------
# Opponent-distribution shaping (E): honesty bias
# ---------------------------------------------------------------------------

def _bet_is_truthful(
    action_id: int,
    actor_hand: list,
    common_hand: list,
    deck_size: int,
    num_jokers: int,
) -> bool:
    """Return True iff the announced set actually exists in the actor's hand
    (plus common cards). Subjective truthfulness: judged from the actor's
    perspective, not the full game's. This is what the actor "knows" when
    choosing their own bet — perfect honesty from their seat.

    Wrapped in a try so a malformed action_id can't break the training loop.
    """
    try:
        cards = list(actor_hand or []) + list(common_hand or [])
        return bool(determine_set_existence(
            cards, int(action_id),
            {"deck_size": int(deck_size)}, int(num_jokers or 0)))
    except Exception:
        return False


def maybe_apply_opponent_honesty_bias(env, legal_mask, action, rng):
    """Trainer-side hook for mechanism E. Dispatches on the personality
    spec's opponent_*_bias fields. Despite the name (kept for backward
    compat with existing callers and tests), this handles all three
    E-family biases:

      - opponent_honesty_bias    -> resample_to_honest      (trust trait)
      - opponent_aggression_bias -> resample_to_aggressive  (defensive trait)
      - opponent_passivity_bias  -> resample_to_passive     (pressurer trait)

    Only the highest-priority active knob (in spec validation order) fires.
    Per spec validation, at most one can be active at a time anyway.

    The learner's own action is NEVER resampled. No-op when no learning
    seat is designated or no E knob is active. Safe under exceptions.
    """
    try:
        ref = getattr(env, "_ref_nick", None)
        if ref is None:
            return int(action)
        cp = env.game.get("cp_nickname")
        if cp == ref:
            return int(action)
        spec = getattr(env.deck_spec, "personality_spec", None)
        if spec is None:
            return int(action)
        honesty = float(getattr(spec, "opponent_honesty_bias", 0.0) or 0.0)
        aggression = float(getattr(spec, "opponent_aggression_bias", 0.0) or 0.0)
        passivity = float(getattr(spec, "opponent_passivity_bias", 0.0) or 0.0)
        if max(honesty, aggression, passivity) <= 0.0:
            return int(action)
        check_id = int(env.deck_spec.check_action_id)
        if honesty > 0.0:
            actor_hand = []
            for h in env.game.get("hands") or []:
                if h.get("nickname") == cp:
                    actor_hand = h.get("hand") or []
                    break
            common = env.game.get("common_hand") or []
            rules = env.game.get("rules") or {}
            return resample_to_honest(
                intended_action=int(action), legal_mask=legal_mask,
                actor_hand=actor_hand, common_hand=common,
                deck_size=int(rules.get("deck_size", 24)),
                num_jokers=int(rules.get("jokers", 0)),
                check_action_id=check_id,
                honesty_bias=honesty, rng=rng,
            )
        if aggression > 0.0:
            return resample_to_aggressive(
                intended_action=int(action), legal_mask=legal_mask,
                check_action_id=check_id,
                aggression_bias=aggression, rng=rng,
            )
        if passivity > 0.0:
            return resample_to_passive(
                intended_action=int(action), legal_mask=legal_mask,
                check_action_id=check_id,
                passivity_bias=passivity, rng=rng,
            )
        return int(action)
    except Exception:
        return int(action)


def resample_to_honest(
    intended_action: int,
    legal_mask,
    *,
    actor_hand: list,
    common_hand: list,
    deck_size: int,
    num_jokers: int,
    check_action_id: int,
    honesty_bias: float,
    rng,
) -> int:
    """Given the opponent's intended action, with probability
    ``honesty_bias`` resample to a truthful alternative — provided one is
    legal. If the intended action was already truthful, or no truthful
    alternative is legal, returns the intended action unchanged.

    CHECK (action_id == check_action_id) is never altered: it isn't a
    truth-bearing claim, just an end-of-round signal.

    The resampler picks uniformly from the truthful-and-legal set; we don't
    try to preserve the policy's relative weighting within it. The
    intuition: under high honesty bias we want a strong distributional
    shift, not a softer nudge. A future refinement could weight by the
    policy's raw logits restricted to the truthful set, but that's a
    second-order knob.

    Args:
      intended_action: the action the opponent's policy actually sampled.
      legal_mask: torch tensor or numpy array of {0, 1} over the full
                  action space; only entries equal to 1 are eligible.
      actor_hand: list of {"value", "colour"} dicts for the OPPONENT's
                  visible cards (their own hand from their POV).
      common_hand: shared common cards (visible to all).
      deck_size, num_jokers: rules.
      check_action_id: the integer id for CHECK.
      honesty_bias: float in [0, 1]. 0 => never resample.
      rng: a random.Random or numpy.random.Generator with .random() returning
           [0, 1) and .choice(seq) returning one element. Either works because
           we only call .random() and .choice().

    Returns the (possibly resampled) action_id as int.
    """
    a = int(intended_action)
    if honesty_bias <= 0.0:
        return a
    if a == int(check_action_id):
        return a
    # Already truthful? Leave it.
    if _bet_is_truthful(a, actor_hand, common_hand, deck_size, 0
                        if num_jokers is None else int(num_jokers)):
        return a
    # The intended action is a bluff. With prob honesty_bias, replace.
    try:
        if rng.random() >= float(honesty_bias):
            return a
    except Exception:
        return a
    # Build the set of legal truthful actions (excluding CHECK).
    if hasattr(legal_mask, "detach"):
        mask_arr = legal_mask.detach().cpu().numpy()
    else:
        mask_arr = np.asarray(legal_mask)
    truthful: list = []
    for aid in range(int(check_action_id)):
        if mask_arr[aid] <= 0:
            continue
        if _bet_is_truthful(aid, actor_hand, common_hand, deck_size,
                            0 if num_jokers is None else int(num_jokers)):
            truthful.append(aid)
    if not truthful:
        return a   # No truthful alternative; the opponent's "honest" play is
                   # to either CHECK (handled upstream) or bluff anyway.
    try:
        return int(rng.choice(truthful))
    except Exception:
        return a


def resample_to_aggressive(
    intended_action: int,
    legal_mask,
    *,
    check_action_id: int,
    aggression_bias: float,
    rng,
) -> int:
    """With probability ``aggression_bias``, replace the opponent's intended
    action with the HIGHEST legal bet (max-escalation). CHECK is never
    overridden (CHECK is the round-ending signal, not an escalation choice).

    Used in mechanism E to train a bot against an always-escalating
    opponent — the bot's optimal response is patient defence (CHECK more
    often, don't overcommit, let escalators tire themselves out). At
    inference vs an actual passive / Conservative opponent, this trait
    presents as the bot being measurably more defensive than baseline.

    If the intended action is already the max legal bet (or if no bets are
    legal at all, only CHECK), the intended action is returned unchanged.
    """
    a = int(intended_action)
    if aggression_bias <= 0.0 or a == int(check_action_id):
        return a
    try:
        if rng.random() >= float(aggression_bias):
            return a
    except Exception:
        return a
    if hasattr(legal_mask, "detach"):
        mask_arr = legal_mask.detach().cpu().numpy()
    else:
        mask_arr = np.asarray(legal_mask)
    max_bet = -1
    for aid in range(int(check_action_id) - 1, -1, -1):
        if mask_arr[aid] > 0:
            max_bet = aid
            break
    if max_bet < 0 or max_bet == a:
        return a
    return int(max_bet)


def resample_to_passive(
    intended_action: int,
    legal_mask,
    *,
    check_action_id: int,
    passivity_bias: float,
    rng,
) -> int:
    """With probability ``passivity_bias``, replace the opponent's intended
    action with CHECK if legal, else with the LOWEST legal bet.

    Used in mechanism E to train a bot against a perpetually-passive
    opponent — the bot's optimal response is to press hard: open
    aggressively, escalate, treat opponents as foldable. At inference
    against an actual non-passive opponent, this trait presents as the bot
    being measurably more aggressive than baseline (higher opening bets,
    lower CHECK rate, more bluffs).

    If the intended action is already CHECK (or CHECK is legal and that's
    what we'd resample to), returns the intended action unchanged.
    """
    a = int(intended_action)
    if passivity_bias <= 0.0 or a == int(check_action_id):
        return a
    try:
        if rng.random() >= float(passivity_bias):
            return a
    except Exception:
        return a
    if hasattr(legal_mask, "detach"):
        mask_arr = legal_mask.detach().cpu().numpy()
    else:
        mask_arr = np.asarray(legal_mask)
    # Prefer CHECK if legal.
    if mask_arr[int(check_action_id)] > 0:
        return int(check_action_id)
    # Else lowest legal bet.
    for aid in range(int(check_action_id)):
        if mask_arr[aid] > 0:
            return int(aid)
    return a


# ---------------------------------------------------------------------------
# Action-mask restriction (B)
# ---------------------------------------------------------------------------

_HAND_TYPE_ACTION_IDS_CACHE: dict = {}


def _action_ids_for_hand_type(set_type: str, deck_size: int, check_id: int) -> list:
    """Action ids in ``[0, check_id)`` whose decoded set_type matches.

    Cached by (set_type, deck_size, check_id) — the partition is static for a
    given deck.
    """
    key = (set_type, int(deck_size), int(check_id))
    cached = _HAND_TYPE_ACTION_IDS_CACHE.get(key)
    if cached is not None:
        return cached
    ids: list = []
    for aid in range(check_id):
        info = get_set_details_from_action_id(aid, deck_size)
        if info and info.get("set_type") == set_type:
            ids.append(aid)
    _HAND_TYPE_ACTION_IDS_CACHE[key] = ids
    return ids


def apply_personality_to_mask(
    mask: np.ndarray,
    spec: Optional[PersonalityTrainSpec],
    deck_spec,
) -> np.ndarray:
    """Apply hard mask restrictions with a mandatory safety rail.

    Order:
      1. Snapshot the original mask and whether CHECK was originally legal.
      2. Zero all bet ids matching ``spec.forbid_hand_types``.
      3. If ``spec.forbid_check_unless_only`` and any bet is still legal, zero CHECK.
      4. If the mask is now all zero, restore CHECK (if originally legal) or
         fall back to the original mask. Never produce an empty mask.
    """
    if spec is None or (not spec.forbid_hand_types and not spec.forbid_check_unless_only):
        return mask
    check_id = int(deck_spec.check_action_id)
    deck_size = int(deck_spec.deck_size)

    original = mask.copy()
    check_legal = bool(mask[check_id]) if check_id < len(mask) else False

    for ht in spec.forbid_hand_types:
        for aid in _action_ids_for_hand_type(ht, deck_size, check_id):
            if aid < len(mask):
                mask[aid] = 0

    if spec.forbid_check_unless_only and check_id < len(mask):
        bets_legal = bool(mask[:check_id].any()) if check_id > 0 else False
        if bets_legal:
            mask[check_id] = 0

    if not mask.any():
        if check_legal:
            mask[check_id] = 1
        else:
            mask[:] = original
    return mask


# ---------------------------------------------------------------------------
# Terminal-magnitude shaping (A)
# ---------------------------------------------------------------------------

def compute_personality_terminal_scale(
    spec: Optional[PersonalityTrainSpec],
    game_state: dict,
    actor_nick: str,
    loser_nick: Optional[str],
    deck_size: int,
) -> float:
    """Compute the scalar multiplied into the MC terminal credit.

    Returns 1.0 when no shaping applies. Mechanics:

      * ``win_multiplier`` / ``loss_multiplier`` set the BASE scale.
      * ``bluff_caught_penalty`` adds to the loss scale if the actor was the
        bet-maker AND the bet was a bluff (the bet-maker lost the round).
      * ``bluff_call_bonus`` adds to the win scale if the actor caught a bluff
        (the actor was NOT the bet-maker and the bet-maker lost).
      * ``hand_type_bias`` adds ``+bias`` to the scale per actor-bet of that
        set_type when the actor won, and ``-bias`` when the actor lost — so
        positive bias = the personality "likes" the hand type (wins bigger,
        loses smaller) and negative = "dislikes" (wins smaller, loses bigger).
    """
    if spec is None or loser_nick is None:
        return 1.0

    actor_won = loser_nick != actor_nick
    scale = float(spec.win_multiplier if actor_won else spec.loss_multiplier)

    check_id = GameRules(int(deck_size)).check_action_id
    history = game_state.get("history") or []

    # Find the last bet and its maker.
    bet_maker_nick: Optional[str] = None
    last_bet_id = -1
    for ev in reversed(history):
        try:
            aid = int(ev.get("action_id", -1))
        except Exception:
            continue
        if 0 <= aid < check_id:
            last_bet_id = aid
            bet_maker_nick = ev.get("player")
            break

    bet_was_bluff = bet_maker_nick is not None and bet_maker_nick == loser_nick

    if bet_was_bluff:
        actor_was_bet_maker = bet_maker_nick == actor_nick
        if actor_won and not actor_was_bet_maker:
            scale += float(spec.bluff_call_bonus)
        if (not actor_won) and actor_was_bet_maker:
            scale += float(spec.bluff_caught_penalty)

    if spec.hand_type_bias:
        for ev in history:
            if ev.get("player") != actor_nick:
                continue
            try:
                aid = int(ev.get("action_id", -1))
            except Exception:
                continue
            if not (0 <= aid < check_id):
                continue
            info = get_set_details_from_action_id(aid, int(deck_size))
            if not info:
                continue
            bias = float(spec.hand_type_bias.get(info.get("set_type"), 0.0))
            scale += bias if actor_won else -bias

    return float(scale)
