"""Smoke test for S7 outputs against the checked-in processed data."""

import sys
import unittest
from html.parser import HTMLParser
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fnp.visualize import FIGURE_NAMES, SIMPLE_FIGURE_NAMES, run_visualize


ROOT = Path(__file__).resolve().parents[1]
PAIR_FILE = ROOT / "data" / "processed" / "pair_table.csv"


@unittest.skipUnless(PAIR_FILE.exists(), "real processed pair data is unavailable")
class TestVisualize(unittest.TestCase):
    def test_real_data_figures_exist_and_are_nonempty(self):
        result = run_visualize()
        self.assertEqual(result["status"], "success")
        figures_dir = ROOT / "reports" / "figures"
        for name in FIGURE_NAMES:
            output = figures_dir / name
            self.assertTrue(output.exists(), name)
            self.assertGreater(output.stat().st_size, 0, name)

    def test_dashboard_is_plain_language_and_self_contained(self):
        result = run_visualize()
        self.assertEqual(result["status"], "success")
        dashboard = ROOT / "reports" / "dashboard.html"
        content = dashboard.read_text(encoding="utf-8")

        class StructureParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.depth = 0
                self.visible = []
                self.errors = []
                self.stack = []

            def handle_starttag(self, tag, attrs):
                if tag not in {"meta", "img", "br", "hr", "input", "link"}:
                    self.stack.append(tag)
                if tag == "details":
                    self.depth += 1

            def handle_endtag(self, tag):
                if not self.stack or self.stack[-1] != tag:
                    self.errors.append(f"unexpected closing tag: {tag}")
                else:
                    self.stack.pop()
                if tag == "details":
                    self.depth -= 1
                    if self.depth < 0:
                        self.errors.append("unbalanced details")

            def handle_data(self, data):
                if self.depth == 0:
                    self.visible.append(data)

        parser = StructureParser()
        parser.feed(content)
        parser.close()
        self.assertFalse(parser.errors)
        self.assertFalse(parser.stack)
        self.assertNotIn("http://", content)
        self.assertNotIn("https://", content)
        visible_text = " ".join(parser.visible)
        for banned in ("rho", "Spearman", "p-value", "bootstrap", "permutation", "N_s", "C_main"):
            self.assertNotIn(banned, visible_text)
        results = json.loads((ROOT / "reports" / "results.json").read_text(encoding="utf-8"))
        expected_a1 = f"{results['analyses']['A1_correlation_all']['spearman_rho']:.2f}"
        self.assertIn(expected_a1, visible_text)
        for name in SIMPLE_FIGURE_NAMES:
            output = ROOT / "reports" / "figures_simple" / name
            self.assertTrue(output.exists(), name)
            self.assertGreater(output.stat().st_size, 0, name)


if __name__ == "__main__":
    unittest.main()