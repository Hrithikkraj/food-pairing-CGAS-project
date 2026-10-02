from pathlib import Path
import unittest

import yaml


class TestConfig(unittest.TestCase):
    def test_params_yaml_sensitivity_cutoffs_and_iterations(self):
        config_path = Path(__file__).parents[1] / "config" / "params.yaml"
        with config_path.open(encoding="utf-8") as config_file:
            params = yaml.safe_load(config_file)

        for cutoff in params["sensitivity_cutoffs"]:
            self.assertEqual(len(cutoff), 2)
            self.assertLess(cutoff[0], cutoff[1])

        self.assertIs(type(params["n_boot"]), int)
        self.assertGreater(params["n_boot"], 0)
        self.assertIs(type(params["n_perm"]), int)
        self.assertGreater(params["n_perm"], 0)


if __name__ == "__main__":
    unittest.main()