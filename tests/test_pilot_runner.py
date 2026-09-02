"""Unit tests for Phase 2 Pilot Benchmark Runner, Manifest Validation, and Resumability."""

import os
import json
import tempfile
import unittest
import numpy as np

from src.training.run_pilot import (
    validate_provenance_record,
    check_run_already_completed,
    audit_manifest_progress,
)


class TestPilotRunner(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
        self.manifest_path = os.path.abspath("04_EXPERIMENTS/pilot_run_manifest.json")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_official_manifest_integrity_84_runs(self):
        """Verify that the official manifest exists and contains exactly 84 unique runs."""
        self.assertTrue(os.path.exists(self.manifest_path), "Manifest file must exist.")
        with open(self.manifest_path, "r") as f:
            manifest = json.load(f)

        self.assertEqual(manifest.get("total_runs"), 84)
        runs = manifest.get("runs", [])
        self.assertEqual(len(runs), 84)

        # Check uniqueness of experiment IDs
        exp_ids = [r["experiment_id"] for r in runs]
        self.assertEqual(len(exp_ids), len(set(exp_ids)), "All 84 experiment IDs must be unique.")

        # Check required fields
        required_fields = ["experiment_id", "dataset", "model", "noise_regime", "track", "seed", "epochs"]
        for r in runs:
            for field in required_fields:
                self.assertIn(field, r, f"Run {r.get('experiment_id')} missing {field}")

    def test_validate_provenance_record_schema(self):
        """Verify that validate_provenance_record correctly validates complete vs corrupted records."""
        valid_record = {
            "status": "COMPLETED",
            "provenance": {
                "experiment_id": "TEST-01",
                "timestamp_utc": "2026-09-02T12:00:00Z",
                "git_commit_sha": "abcdef123456",
            },
            "configuration": {"track": "CE", "seed": 42},
            "metrics": {
                "test_top1_acc": 50.4,
                "raw_test_ece": 0.05,
                "raw_ada_ece": 0.052,
                "raw_brier": 0.61,
                "ts_clean_test_ece": 0.045,
                "ts_corrupted_test_ece": 0.048,
                "val_calibration_delta": 0.003,
            },
            "history": {"train_loss": [1.5], "train_acc": [40.0], "clean_val_acc": [50.0]},
        }
        self.assertTrue(validate_provenance_record(valid_record))

        # Failed status
        failed_record = valid_record.copy()
        failed_record["status"] = "FAILED"
        self.assertFalse(validate_provenance_record(failed_record))

        # Missing metric
        corrupted_record = json.loads(json.dumps(valid_record))
        corrupted_record["metrics"]["raw_test_ece"] = None
        self.assertFalse(validate_provenance_record(corrupted_record))

        # Non-finite metric
        nan_record = json.loads(json.dumps(valid_record))
        nan_record["metrics"]["raw_test_ece"] = float("nan")
        self.assertFalse(validate_provenance_record(nan_record))

    def test_check_run_already_completed_and_resumability(self):
        """Verify that completed valid runs are detected and never overwritten."""
        exp_id = "PILOT-TEST-EXP-01"
        self.assertFalse(check_run_already_completed(self.output_dir, exp_id))

        # Write valid completed record
        valid_record = {
            "status": "COMPLETED",
            "provenance": {
                "experiment_id": exp_id,
                "timestamp_utc": "2026-09-02T12:00:00Z",
                "git_commit_sha": "abcdef123456",
            },
            "configuration": {"track": "CE"},
            "metrics": {
                "test_top1_acc": 50.4,
                "raw_test_ece": 0.05,
                "raw_ada_ece": 0.052,
                "raw_brier": 0.61,
                "ts_clean_test_ece": 0.045,
                "ts_corrupted_test_ece": 0.048,
                "val_calibration_delta": 0.003,
            },
            "history": {"train_loss": [1.5]},
        }
        out_file = os.path.join(self.output_dir, f"{exp_id}.json")
        with open(out_file, "w") as f:
            json.dump(valid_record, f)

        self.assertTrue(check_run_already_completed(self.output_dir, exp_id))

    def test_audit_manifest_progress(self):
        """Verify audit_manifest_progress accounts for all 84 official runs."""
        audit = audit_manifest_progress(self.manifest_path, self.output_dir)
        self.assertEqual(audit["total_manifest_runs"], 84)
        self.assertEqual(audit["completed_runs"], 0)
        self.assertEqual(audit["missing_runs"], 84)

        # Simulate 1 completed run
        with open(self.manifest_path, "r") as f:
            manifest = json.load(f)
        first_exp_id = manifest["runs"][0]["experiment_id"]

        valid_record = {
            "status": "COMPLETED",
            "provenance": {
                "experiment_id": first_exp_id,
                "timestamp_utc": "2026-09-02T12:00:00Z",
                "git_commit_sha": "abcdef123456",
            },
            "configuration": {},
            "metrics": {
                "test_top1_acc": 50.4,
                "raw_test_ece": 0.05,
                "raw_ada_ece": 0.052,
                "raw_brier": 0.61,
                "ts_clean_test_ece": 0.045,
                "ts_corrupted_test_ece": 0.048,
                "val_calibration_delta": 0.003,
            },
            "history": {},
        }
        with open(os.path.join(self.output_dir, f"{first_exp_id}.json"), "w") as f:
            json.dump(valid_record, f)

        audit2 = audit_manifest_progress(self.manifest_path, self.output_dir)
        self.assertEqual(audit2["completed_runs"], 1)
        self.assertEqual(audit2["missing_runs"], 83)


if __name__ == "__main__":
    unittest.main()
