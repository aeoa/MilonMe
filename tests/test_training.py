import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import requests

from milon_api import download_all
from training_data import PlotContext, merge_history, migrate_cache, read_json, write_json
from training_plots import PLOT_GROUPS, aggregate_frame, plot_all


class HistoryTests(unittest.TestCase):
    def test_bioage_rollover_and_revision(self):
        old = [{"t": 1, "bioAge": 30}, {"t": 2, "bioAge": 31}]
        new = [{"t": 2, "bioAge": 30.5}, {"t": 3, "bioAge": 32}]
        merged = merge_history("bioage_history", old, new)
        self.assertEqual([r["t"] for r in merged], [1, 2, 3])
        self.assertEqual(merged[1]["bioAge"], 30.5)
        self.assertEqual(merge_history("bioage_history", merged, new), merged)
        self.assertEqual(merge_history("bioage_history", merged, []), merged)

    def test_device_and_circuit_keys_do_not_collapse(self):
        def row(date, device, kind="kz", weight=30):
            return {"datepart": date, "id": device, "nodegroup_type": kind, "aw": weight}
        old = {"devices_weeks": [], "devices_months": [row("2024-07-01", 1), row("2025-07-01", 1)]}
        new = {"devices_weeks": [], "devices_months": [row("2025-07-01", 1, weight=40),
                row("2025-07-01", 2), row("2025-07-01", 1, "ka")]}
        merged = merge_history("progress-devices", old, new)
        self.assertEqual(len(merged["devices_months"]), 4)
        self.assertEqual([r["aw"] for r in merged["devices_months"] if r["id"] == 1
                          and r["nodegroup_type"] == "kz"], [30, 40])

    def test_circuit_rollover_and_empty_response(self):
        old = {"devices": [1], "nodegroups_weeks": [{"datepart": "2024-07-01", "nodegroup_type": "kz"}],
               "nodegroups_months": []}
        new = {"devices": [2], "nodegroups_weeks": [], "nodegroups_months": []}
        merged = merge_history("progress", old, new)
        self.assertEqual(merged["devices"], [1, 2])
        self.assertEqual(merged["nodegroups_weeks"], old["nodegroups_weeks"])

    def test_monthly_buckets_survive_daylight_saving(self):
        rows = [{"datepart": f"2026-{month:02d}-01T00:00:00.000Z", "aw": month} for month in range(1, 10)]
        frame = aggregate_frame(rows, pd.Timestamp("2026-02-24", tz="Europe/Berlin"))
        table = frame.set_index("date").reindex(pd.date_range("2026-02-01", "2026-09-01", freq="MS"))
        self.assertEqual(len(table), 8)
        self.assertFalse(table["aw"].isna().any())

    def test_download_preserves_cache_on_failure_and_accumulates_on_success(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "test-user" / "stats"
            write_json(folder / "bioage_history.json", [{"t": 1, "bioAge": 30}])
            write_json(folder / "snapshots" / "2000-01-01" / "muscles.json", {"data": {"old": True}})
            session = Mock()

            def get(url, timeout):
                response = Mock()
                if "/bioage_history/" in url:
                    response.json.return_value = [{"t": 2, "bioAge": 31}]
                elif "/muscles/" in url:
                    response.json.return_value = {"premium": {}}
                else:
                    raise requests.Timeout()
                return response

            session.get.side_effect = get
            with contextlib.redirect_stdout(io.StringIO()):
                download_all(session, "test-user", "test-studio", directory)
            self.assertEqual([r["t"] for r in read_json(folder / "bioage_history.json")], [1, 2])
            self.assertEqual(len(list((folder / "snapshots").glob("*/muscles.json"))), 2)
            session.get.side_effect = requests.Timeout()
            with contextlib.redirect_stdout(io.StringIO()):
                download_all(session, "test-user", "test-studio", directory)
            self.assertEqual(len(read_json(folder / "bioage_history.json")), 2)
            self.assertFalse(read_json(folder / "manifest.json")["endpoints"]["bioage_history"]["ok"])

    def test_malformed_history_is_rejected(self):
        with self.assertRaises(ValueError):
            merge_history("bioage_history", [], {"error": "unavailable"})
        with self.assertRaises(ValueError):
            merge_history("progress", {}, {"devices": []})

    def test_empty_cache_removes_stale_optional_plots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "graphs" / "test-user"
            output.mkdir(parents=True)
            (output / "heart_rate.png").write_bytes(b"stale")
            with contextlib.redirect_stdout(io.StringIO()):
                plot_all(PlotContext.load("test-user", root / "data", root / "graphs", 1))
            self.assertFalse((output / "heart_rate.png").exists())

    def test_migration_preserves_history_and_snapshots_and_is_repeatable(self):
        with tempfile.TemporaryDirectory() as directory:
            stats = Path(directory)
            old = stats / "insights"
            write_json(old / "bioage_history.json", [{"t": 1, "bioAge": 30}, {"t": 2, "bioAge": 31}])
            write_json(stats / "bioage_history.json", [{"t": 2, "bioAge": 32}])
            snapshot = {"captured_at": "2026-09-24", "data": {"id": 7}}
            relative = Path("snapshots/2026-09-24/plans/7.json")
            write_json(old / relative, snapshot)
            migrate_cache(stats)
            migrate_cache(stats)
            self.assertEqual(read_json(stats / "bioage_history.json"), [{"t": 1, "bioAge": 30}, {"t": 2, "bioAge": 32}])
            self.assertEqual(read_json(stats / relative), snapshot)
            self.assertTrue((old / relative).exists())

    def test_plot_groups_share_context_and_only_clear_selected_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "graphs" / "user"
            output.mkdir(parents=True)
            (output / "weights.png").write_bytes(b"stale")
            (output / "bioage.png").write_bytes(b"stale")
            (output / "heart_rate.png").write_bytes(b"keep")
            strength, bioage = Mock(), Mock()
            selected = {"strength": (strength, ("weights.png",)), "bioage": (bioage, ("bioage.png",))}
            with patch.dict(PLOT_GROUPS, selected), contextlib.redirect_stdout(io.StringIO()):
                plot_all(PlotContext.load("user", root / "data", root / "graphs", 28), ["strength", "bioage"])
            self.assertIs(strength.call_args.args[0], bioage.call_args.args[0])
            self.assertEqual(strength.call_args.args[0].days, 28)
            self.assertFalse((output / "weights.png").exists())
            self.assertFalse((output / "bioage.png").exists())
            self.assertEqual((output / "heart_rate.png").read_bytes(), b"keep")

    def test_download_premium_and_other_sources_fail_independently(self):
        with tempfile.TemporaryDirectory() as directory:
            stats = Path(directory) / "user" / "stats"
            context = PlotContext.load("user", directory, Path(directory) / "graphs")
            month = context.now.strftime("%y%m")
            previous = {"stats": []}
            write_json(stats / "premium" / f"{month}.json", previous)
            session = Mock()

            def get(url, timeout):
                if "/premium/" in url:
                    raise requests.Timeout()
                response = Mock()
                response.json.return_value = [] if "/bioage_history/" in url else {}
                return response

            session.get.side_effect = get
            with contextlib.redirect_stdout(io.StringIO()):
                download_all(session, "user", "studio", directory)
            self.assertEqual(read_json(stats / "premium" / f"{month}.json"), previous)
            manifest = read_json(stats / "manifest.json")["endpoints"]
            self.assertFalse(manifest[f"premium/{month}"]["ok"])
            self.assertTrue(manifest["bioage_history"]["ok"])


if __name__ == "__main__":
    unittest.main()
