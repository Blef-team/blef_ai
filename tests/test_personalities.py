"""Unit + smoke tests for the personality layer (nfsp_ai/personalities.py)."""

import os
import unittest

import numpy as np
import torch

from nfsp_ai import personalities as P
from shared.game_utils import GameRules


# --------------------------------------------------------------------------- #
# Pure-function tests (no model needed)
# --------------------------------------------------------------------------- #
class SculptTest(unittest.TestCase):
    def _cfg(self, **kw):
        base = dict(source="pi", risk=0.0, guard=0.0, susp=0.0, tempo=0.0,
                    chaos=0.0, tau_model=1.0, sample=False)
        base.update(kw)
        return P.EffectiveCfg(**base)

    def _legal_mask(self, A, legal_idx):
        m = torch.zeros(A)
        for i in legal_idx:
            m[i] = 1.0
        return m

    def test_illegal_actions_stay_neg_inf(self):
        A, check = 89, 88
        logits = torch.randn(1, A)
        legal = [3, 4, 5, 60, check]      # a few bets + check
        mask = self._legal_mask(A, legal)
        pvt = [0.2] * check
        pub = [0.1] * check
        out = P.sculpt_logits(logits, mask, self._cfg(risk=1.5, susp=1.0, tempo=1.0),
                              pvt=pvt, pub=pub, p_last=0.2, check_id=check)
        illegal = [i for i in range(A) if i not in legal]
        self.assertTrue(torch.isinf(out[0, illegal]).all() or
                        (out[0, illegal] <= torch.finfo(out.dtype).min / 2).all())
        # softmax puts ~zero mass on illegal and the chosen action is legal
        probs = torch.softmax(out, dim=-1)[0]
        self.assertLess(float(probs[illegal].sum()), 1e-6)
        self.assertIn(int(out.argmax(-1).item()), legal)

    def test_risk_sign_makes_bluffier(self):
        # Higher risk should shift mass toward lower-pPriv (improbable) bets.
        A, check = 89, 88
        logits = torch.zeros(1, A)        # flat policy isolates the overlay
        legal_idx = list(range(10, 40))
        mask = self._legal_mask(A, legal_idx)
        # pvt decreasing with index (senior bets less probable); pub flat
        pvt = [0.0] * check
        for r, i in enumerate(legal_idx):
            pvt[i] = 1.0 - r / len(legal_idx)
        pub = [0.3] * check
        honest = P.sculpt_logits(logits, mask, self._cfg(risk=-1.5),
                                 pvt=pvt, pub=pub, p_last=0.2, check_id=check)
        bluffy = P.sculpt_logits(logits, mask, self._cfg(risk=+1.5),
                                 pvt=pvt, pub=pub, p_last=0.2, check_id=check)
        # expected pPriv of the argmax bet
        self.assertGreater(pvt[int(honest.argmax(-1))], pvt[int(bluffy.argmax(-1))])

    def test_susp_boosts_check_when_implausible(self):
        A, check = 89, 88
        logits = torch.zeros(1, A)
        legal_idx = [20, 21, 22, check]
        mask = self._legal_mask(A, legal_idx)
        pvt = [0.5] * check
        pub = [0.5] * check
        # last bet very implausible (p_last low) + paranoid -> check boosted
        out = P.sculpt_logits(logits, mask, self._cfg(susp=2.0),
                              pvt=pvt, pub=pub, p_last=0.05, check_id=check)
        self.assertEqual(int(out.argmax(-1).item()), check)
        # trusting bot with same state should not jump to check
        out2 = P.sculpt_logits(logits, mask, self._cfg(susp=-2.0),
                               pvt=pvt, pub=pub, p_last=0.05, check_id=check)
        self.assertNotEqual(int(out2.argmax(-1).item()), check)


class SelectTest(unittest.TestCase):
    def test_argmax_when_no_chaos(self):
        logits = torch.tensor([[0.1, 5.0, 0.2, float("-inf")]])
        self.assertEqual(P._select(logits, chaos=0.0), 1)

    def test_sampling_stays_in_distribution(self):
        torch.manual_seed(0)
        # one strong + a few weak legal; chaos should sample only among top-k
        logits = torch.tensor([[3.0, 2.5, 2.0, -5.0, float("-inf")]])
        picks = {P._select(logits, chaos=1.0) for _ in range(200)}
        self.assertTrue(picks.issubset({0, 1, 2}))   # never the clearly-bad 3 or illegal 4
        self.assertNotIn(4, picks)


class ResolveTest(unittest.TestCase):
    def _state(self, cp, players_extra=None):
        players = [{"nickname": cp, "n_cards": 1}]
        if players_extra:
            players[0].update(players_extra)
        return {"cp_nickname": cp, "players": players}

    def test_explicit_field(self):
        s = self._state("whoever", {"personality": "Veles"})
        self.assertEqual(P.resolve_personality(s), "veles")

    def test_nickname_fallback(self):
        self.assertEqual(P.resolve_personality(self._state("perun_(AI)")), "perun")
        self.assertEqual(P.resolve_personality(self._state("veles_2_(AI)")), "veles")

    def test_unknown_returns_none(self):
        self.assertIsNone(P.resolve_personality(self._state("p0")))
        self.assertIsNone(P.resolve_personality(self._state("randomname")))

    def test_every_god_resolvable_and_typed(self):
        self.assertEqual(len(P.PERSONALITIES), 16)
        self.assertNotIn("morana", P.PERSONALITIES)  # CFR, deployed separately
        for name, cfg in P.PERSONALITIES.items():
            self.assertIn(cfg.source, {"pi", "q", "conservative", "conservative_crawling", "cfr"})


class MoodTest(unittest.TestCase):
    def test_phase_ramp_interpolates(self):
        cfg = P.PersonalityConfig(source="pi", mood="phase_ramp",
                                  mood_params={"risk_early": -1.0, "risk_late": 1.0})
        early = P.derive_effective_cfg(cfg, {"depth": 0.0, "standing": 0.0, "parity": 0}, 0.5, 0.5)
        late = P.derive_effective_cfg(cfg, {"depth": 1.0, "standing": 0.0, "parity": 0}, 0.5, 0.5)
        self.assertAlmostEqual(early.risk, -1.0, places=5)
        self.assertAlmostEqual(late.risk, 1.0, places=5)

    def test_parity_swing_flips(self):
        cfg = P.PERSONALITIES["zorya"]
        dawn = P.derive_effective_cfg(cfg, {"depth": 0.3, "standing": 0.0, "parity": 0}, 0.5, 0.5)
        dusk = P.derive_effective_cfg(cfg, {"depth": 0.3, "standing": 0.0, "parity": 1}, 0.5, 0.5)
        self.assertGreater(dawn.risk, dusk.risk)   # dawn aggressive
        self.assertLess(dawn.susp, dusk.susp)       # dusk suspicious

    def test_state_hash_deterministic(self):
        s = {"history": [{"action_id": 3}], "cp_nickname": "x", "round_number": 2}
        self.assertEqual(P._state_hash(s), P._state_hash(dict(s)))


# --------------------------------------------------------------------------- #
# Model smoke tests (load the production agent once)
# --------------------------------------------------------------------------- #
def _have_artifacts():
    import glob
    return bool(glob.glob("artifacts/nfsp_inference_*.pt"))


@unittest.skipUnless(_have_artifacts(), "inference artifacts not present")
class PersonalityActionSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from nfsp_ai import production_agent as pa
        cls.agent = pa.load_agent()

    def _state(self, deck=24, n=2, k=1, last_bet=2, personality=None):
        vals = 6 if deck == 24 else 8
        cards = [(v, c) for v in range(vals) for c in range(4)]
        hands_cards = [cards[i * k:(i + 1) * k] for i in range(n)]
        names = [f"p{i}" for i in range(n)]
        players = [{"nickname": names[i], "n_cards": k, "team": None} for i in range(n)]
        if personality is not None:
            players[0]["personality"] = personality
        hands = [{"nickname": names[i],
                  "hand": [{"value": v, "colour": c} for (v, c) in hands_cards[i]]}
                 for i in range(n)]
        hist = [] if last_bet is None else [{"action_id": last_bet, "player": names[-1]}]
        return {"cp_nickname": names[0],
                "rules": {"deck_size": deck, "jokers": 0, "blanks": 0,
                          "common_cards": 0, "max_rounds": 8},
                "players": players, "hands": hands, "common_hand": [],
                "history": hist, "round_number": 1, "game_uuid": "t"}

    def _assert_legal(self, action, deck, last_bet):
        check = GameRules(deck).check_action_id
        self.assertTrue(0 <= action <= check)
        if last_bet is not None:
            self.assertTrue(action == check or action > last_bet)

    def test_unknown_personality_equals_baseline(self):
        s = self._state(personality=None)
        base = self.agent.determine_action(s)
        s2 = self._state(personality="not_a_real_god")
        self.assertEqual(self.agent.determine_action(s2), base)

    def test_all_personalities_return_legal_actions(self):
        for deck in (24, 32):
            for name in P.PERSONALITY_NAMES:
                s = self._state(deck=deck, n=2, k=2, last_bet=3, personality=name)
                a = self.agent.determine_action(s)
                self._assert_legal(a, deck, 3)

    def test_delegated_domovoi_is_legal(self):
        s = self._state(deck=24, n=4, k=1, last_bet=2, personality="domovoi")
        a = self.agent.determine_action(s)
        self._assert_legal(a, 24, 2)


if __name__ == "__main__":
    unittest.main()
