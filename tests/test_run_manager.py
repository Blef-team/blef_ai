"""Smoke tests for the run-management CLI.

The full subprocess-launch path requires a long-running trainer; here we
exercise the surface area that doesn't need to actually launch nfsp_run_local.
"""

import json
import os
import tempfile
import unittest

from tools import run_manager


class RunDirDiscovery(unittest.TestCase):
    def test_list_run_dirs_filters_by_naming_convention(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Two valid run dirs and one bogus one.
            valid_a = os.path.join(tmp, "20260101-120000__alpha")
            valid_b = os.path.join(tmp, "20260102-130000__beta")
            bogus = os.path.join(tmp, "not-a-run-dir")
            for d in (valid_a, valid_b, bogus):
                os.makedirs(d)
            runs = run_manager._list_run_dirs(tmp)
            self.assertEqual({r["label"] for r in runs}, {"20260101-120000__alpha", "20260102-130000__beta"})
            # Newest first
            self.assertEqual(runs[0]["timestamp"], "20260102-130000")

    def test_find_run_returns_most_recent_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "20260101-120000__exp"))
            os.makedirs(os.path.join(tmp, "20260201-120000__exp"))
            os.makedirs(os.path.join(tmp, "20260301-120000__other"))
            run = run_manager._find_run(tmp, "exp")
            self.assertIsNotNone(run)
            self.assertEqual(run["timestamp"], "20260201-120000")


class MetricsRead(unittest.TestCase):
    def test_read_latest_metrics_returns_last_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = os.path.join(tmp, "20260101-120000__r")
            os.makedirs(run_dir)
            csv_path = os.path.join(run_dir, "metrics.csv")
            with open(csv_path, "w", encoding="utf-8") as fh:
                fh.write("step,avg_reward,win_rate\n")
                fh.write("100,0.1,0.5\n")
                fh.write("200,0.3,0.7\n")
            latest = run_manager._read_latest_metrics(run_dir)
            self.assertIsNotNone(latest)
            self.assertEqual(latest["step"], "200")
            self.assertEqual(latest["win_rate"], "0.7")

    def test_read_latest_metrics_missing_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(run_manager._read_latest_metrics(tmp))

    def test_read_latest_metrics_merges_same_step_rows(self):
        # The trainer writes two rows per step: a training row with q_loss/sl_loss
        # populated and an eval row with avg_reward/win_rate populated but losses
        # 'nan'. Status should merge them so the user sees one combined snapshot.
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = os.path.join(tmp, "20260101-120000__r")
            os.makedirs(run_dir)
            csv_path = os.path.join(run_dir, "metrics.csv")
            with open(csv_path, "w", encoding="utf-8") as fh:
                fh.write("step,avg_reward,win_rate,q_loss,sl_loss\n")
                fh.write("100,0.5,0.6,0.20,1.8\n")
                fh.write("100,0.3,0.55,nan,nan\n")
            latest = run_manager._read_latest_metrics(run_dir)
            self.assertIsNotNone(latest)
            self.assertEqual(latest["step"], "100")
            self.assertEqual(latest["q_loss"], "0.20")
            self.assertEqual(latest["sl_loss"], "1.8")
            self.assertEqual(latest["avg_reward"], "0.3")
            self.assertEqual(latest["win_rate"], "0.55")


class OverrideSubcommand(unittest.TestCase):
    def test_override_writes_to_control_plane_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = os.path.join(tmp, "20260101-120000__exp")
            os.makedirs(run_dir)
            cp_path = os.path.join(run_dir, "control_plane.json")
            with open(cp_path, "w", encoding="utf-8") as fh:
                json.dump({"meta": {"cooldown_steps": 0}, "overrides": {}}, fh)

            argv = [
                "--runs-root", tmp,
                "override",
                "--experiment-name", "exp",
                "--set", "eta=0.05",
                "--set", "epsilon=none",
            ]
            rc = run_manager.main(argv)
            self.assertEqual(rc, 0)
            with open(cp_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            self.assertEqual(payload["overrides"]["eta"], 0.05)
            self.assertIsNone(payload["overrides"]["epsilon"])


if __name__ == "__main__":
    unittest.main()
