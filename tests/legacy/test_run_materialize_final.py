import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from qeu_bundling.runners import run_materialize_final


class RunMaterializeFinalTests(unittest.TestCase):
    def test_main_returns_zero_and_supports_uncapped_env_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)
            output_path = base_dir / "output" / "final_recommendations_by_user.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fake_result = SimpleNamespace(
                path=output_path,
                run_id="run-full",
                user_count=30000,
                profile_count=30000,
            )
            with patch("qeu_bundling.runners.run_materialize_final.get_paths", return_value=SimpleNamespace(project_root=base_dir)), patch(
                "qeu_bundling.runners.run_materialize_final.ensure_layout",
                return_value=None,
            ), patch(
                "qeu_bundling.runners.run_materialize_final._bootstrap_required_artifacts",
                return_value=None,
            ), patch(
                "qeu_bundling.runners.run_materialize_final.materialize_final_recommendations_by_user",
                return_value=fake_result,
            ) as mocked_materialize, patch.dict(
                os.environ,
                {"QEU_FINAL_RECOMMENDATIONS_MAX_USERS": "0"},
                clear=False,
            ), patch(
                "sys.stdout",
                new_callable=io.StringIO,
            ) as stdout:
                rc = run_materialize_final.main()

        self.assertEqual(rc, 0)
        mocked_materialize.assert_called_once()
        kwargs = mocked_materialize.call_args.kwargs
        self.assertIsNone(kwargs["max_users"])
        payload = json.loads(stdout.getvalue().strip())
        self.assertIsNone(payload["max_users"])
        self.assertEqual(payload["selection_mode"], "sorted")

    def test_main_supports_explicit_10_user_test_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)
            output_path = base_dir / "output" / "final_recommendations_by_user.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fake_result = SimpleNamespace(
                path=output_path,
                run_id="run-test-10",
                user_count=10,
                profile_count=10,
            )
            with patch("qeu_bundling.runners.run_materialize_final.get_paths", return_value=SimpleNamespace(project_root=base_dir)), patch(
                "qeu_bundling.runners.run_materialize_final.ensure_layout",
                return_value=None,
            ), patch(
                "qeu_bundling.runners.run_materialize_final._bootstrap_required_artifacts",
                return_value=None,
            ), patch(
                "qeu_bundling.runners.run_materialize_final.materialize_final_recommendations_by_user",
                return_value=fake_result,
            ) as mocked_materialize, patch(
                "sys.stdout",
                new_callable=io.StringIO,
            ) as stdout:
                rc = run_materialize_final.main(max_users=10, random_sample=False)

        self.assertEqual(rc, 0)
        mocked_materialize.assert_called_once()
        kwargs = mocked_materialize.call_args.kwargs
        self.assertEqual(kwargs["max_users"], 10)
        self.assertEqual(kwargs["user_selection_mode"], "sorted")
        payload = json.loads(stdout.getvalue().strip())
        self.assertEqual(payload["max_users"], 10)


if __name__ == "__main__":
    unittest.main()
