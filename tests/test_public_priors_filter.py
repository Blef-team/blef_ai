"""Unit tests for the inference-time public-priors mask filter.

The filter zeros bet actions whose public-prior probability is effectively
zero (provably-impossible sets given visible cards alone). The filter never
touches CHECK and never produces an empty mask.
"""

import numpy as np
import torch
import unittest

from nfsp_ai.production_agent import (
    _filter_mask_by_public_priors,
    _PUB_PRIOR_EPSILON,
)


class PublicPriorsFilterTest(unittest.TestCase):

    def _build_mask(self, n_bets, check_id, legal_indices):
        m = np.zeros(check_id + 1, dtype=np.float32)
        for i in legal_indices:
            m[i] = 1.0
        return torch.from_numpy(m)

    def test_zero_priors_get_masked(self):
        check_id = 5
        # All bets legal; priors say only ids 0,2,4 are possible.
        mask = self._build_mask(5, check_id, [0, 1, 2, 3, 4, 5])
        pub = [0.5, 0.0, 0.3, 0.0, 0.7]
        out = _filter_mask_by_public_priors(mask, pub, check_id)
        self.assertEqual(out[0].item(), 1.0)
        self.assertEqual(out[1].item(), 0.0)  # filtered (impossible)
        self.assertEqual(out[2].item(), 1.0)
        self.assertEqual(out[3].item(), 0.0)  # filtered (impossible)
        self.assertEqual(out[4].item(), 1.0)
        self.assertEqual(out[5].item(), 1.0)  # CHECK preserved

    def test_check_action_never_filtered(self):
        check_id = 3
        # Mask: all bets + CHECK legal.
        mask = self._build_mask(3, check_id, [0, 1, 2, 3])
        # Priors of length check_id (just bets). CHECK has no prior entry.
        pub = [0.0, 0.0, 0.5]
        out = _filter_mask_by_public_priors(mask, pub, check_id)
        self.assertEqual(out[check_id].item(), 1.0)

    def test_priors_longer_than_check_id_ignored(self):
        check_id = 3
        mask = self._build_mask(3, check_id, [0, 1, 2, 3])
        # Pad priors beyond check_id with junk — should be ignored.
        pub = [0.5, 0.5, 0.5, 999.0, -999.0]
        out = _filter_mask_by_public_priors(mask, pub, check_id)
        # Bets stay, CHECK stays.
        self.assertEqual(out[0].item(), 1.0)
        self.assertEqual(out[3].item(), 1.0)

    def test_priors_shorter_than_check_id(self):
        # Defensive: if pub_prior is shorter, only filter the prefix.
        check_id = 5
        mask = self._build_mask(5, check_id, [0, 1, 2, 3, 4, 5])
        pub = [0.0, 0.5, 0.0]  # length 3 of 5
        out = _filter_mask_by_public_priors(mask, pub, check_id)
        self.assertEqual(out[0].item(), 0.0)
        self.assertEqual(out[1].item(), 1.0)
        self.assertEqual(out[2].item(), 0.0)
        # Beyond filter prefix, original mask preserved.
        self.assertEqual(out[3].item(), 1.0)
        self.assertEqual(out[4].item(), 1.0)
        self.assertEqual(out[5].item(), 1.0)

    def test_safety_rail_keeps_mask_when_filter_would_empty_it(self):
        check_id = 3
        # Only one bet legal (id 1), CHECK illegal (e.g. round resolved).
        mask = self._build_mask(3, check_id, [1])
        pub = [0.0, 0.0, 0.0]  # filter would zero everything
        out = _filter_mask_by_public_priors(mask, pub, check_id)
        # Original mask preserved by the safety rail.
        self.assertEqual(out[1].item(), 1.0)
        self.assertEqual(out.sum().item(), 1.0)

    def test_none_pub_prior_passes_through(self):
        check_id = 3
        mask = self._build_mask(3, check_id, [0, 1, 2, 3])
        out = _filter_mask_by_public_priors(mask, None, check_id)
        self.assertTrue(torch.equal(out, mask))

    def test_empty_pub_prior_passes_through(self):
        check_id = 3
        mask = self._build_mask(3, check_id, [0, 1, 2, 3])
        out = _filter_mask_by_public_priors(mask, [], check_id)
        self.assertTrue(torch.equal(out, mask))

    def test_epsilon_threshold(self):
        check_id = 3
        mask = self._build_mask(3, check_id, [0, 1, 2, 3])
        pub = [_PUB_PRIOR_EPSILON / 2, _PUB_PRIOR_EPSILON, _PUB_PRIOR_EPSILON * 10]
        out = _filter_mask_by_public_priors(mask, pub, check_id)
        # 0 < eps/2 < eps -> filtered (strict >).
        self.assertEqual(out[0].item(), 0.0)
        # exactly eps -> filtered (strict >).
        self.assertEqual(out[1].item(), 0.0)
        # > eps -> kept.
        self.assertEqual(out[2].item(), 1.0)

    def test_only_bets_already_illegal_still_filters_correctly(self):
        # Bets [0,1] illegal already; bet 2 legal; CHECK legal.
        # Filter shouldn't accidentally re-enable [0,1].
        check_id = 3
        mask = self._build_mask(3, check_id, [2, 3])
        pub = [0.9, 0.9, 0.9]
        out = _filter_mask_by_public_priors(mask, pub, check_id)
        self.assertEqual(out[0].item(), 0.0)
        self.assertEqual(out[1].item(), 0.0)
        self.assertEqual(out[2].item(), 1.0)
        self.assertEqual(out[3].item(), 1.0)


if __name__ == "__main__":
    unittest.main()
