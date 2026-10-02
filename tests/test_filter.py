"""Regression tests for S2 filtering rules."""

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fnp.filter import DEFAULT_EXCLUSION_RULES


class TestFilterRules(unittest.TestCase):
    def test_exclusion_rules_match_whole_tokens(self):
        excluded = {"gin", "red_wine", "rum", "rose"}
        kept = {"ginger", "cauliflower", "rosemary"}

        for name in excluded:
            self.assertTrue(any(rule(name) for rule, _ in DEFAULT_EXCLUSION_RULES), name)
        for name in kept:
            self.assertFalse(any(rule(name) for rule, _ in DEFAULT_EXCLUSION_RULES), name)


if __name__ == "__main__":
    unittest.main()
