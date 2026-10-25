import os
import runpy
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


CONSTANTS = Path(__file__).parents[1] / "src" / "cfg" / "constants_jailbreak.py"


def load_constants(environment):
    numpy = types.ModuleType("numpy")
    numpy.inf = float("inf")
    torch = types.ModuleType("torch")
    torch.mps = types.SimpleNamespace(is_available=lambda: False)
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    yaml = types.ModuleType("yaml")
    yaml.safe_load = lambda _stream: {}
    modules = {"numpy": numpy, "torch": torch, "yaml": yaml}
    with patch.dict(os.environ, environment, clear=True), patch.dict(sys.modules, modules):
        return runpy.run_path(str(CONSTANTS))


class JailbreakRunControlTests(unittest.TestCase):
    def test_production_population_defaults_are_unchanged(self):
        values = load_constants({})

        self.assertEqual(values["num_generations"], 17)
        self.assertEqual(values["start_population_size"], 24)
        self.assertEqual(values["population_size"], 24)
        self.assertEqual(values["SELECTION_METHOD"], "map_elites")

    def test_environment_can_request_a_small_bootstrap_population(self):
        values = load_constants({
            "LLMGE_NUM_GENERATIONS": "1",
            "LLMGE_START_POPULATION_SIZE": "12",
            "LLMGE_POPULATION_SIZE": "12",
            "JB_SELECTION": "nsga2",
        })

        self.assertEqual(values["num_generations"], 1)
        self.assertEqual(values["start_population_size"], 12)
        self.assertEqual(values["population_size"], 12)
        self.assertEqual(values["SELECTION_METHOD"], "nsga2")


if __name__ == "__main__":
    unittest.main()
