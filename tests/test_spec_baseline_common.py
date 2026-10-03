import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generation.spec_baseline_common import load_dataset, spec_messages, unsafe_reason


class SpecBaselineCommonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks, cls.x_map, cls.y_map = load_dataset(ROOT / "data" / "generation")

    def test_dataset_contract(self):
        self.assertEqual(len(self.tasks), 762)
        self.assertEqual(len(self.x_map), 762)
        self.assertEqual(len(self.y_map), 762)
        self.assertTrue(all(len(task.shot_ids) == 5 for task in self.tasks))

    def test_zero_shot_prompt_has_only_system_and_target(self):
        messages = spec_messages("TARGET_X")
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertIn("TARGET_X", messages[-1]["content"])

    def test_five_shot_prompt_preserves_order(self):
        examples = [(f"SHOT_X_{i}", f"SHOT_Y_{i}") for i in range(5)]
        messages = spec_messages("TARGET_X", examples)
        self.assertEqual(len(messages), 12)
        assistant_outputs = [m["content"] for m in messages if m["role"] == "assistant"]
        self.assertEqual(len(assistant_outputs), 5)
        for index, output in enumerate(assistant_outputs):
            self.assertIn(f"SHOT_Y_{index}", output)
        self.assertNotIn("SHOT_Y_", messages[-1]["content"])

    def test_unsafe_constructs_are_rejected(self):
        valid = "verus! { fn f() -> (r: bool) ensures r { true } }"
        self.assertIsNone(unsafe_reason(valid))
        self.assertIsNotNone(unsafe_reason(valid.replace("true }", "assume(true); true }")))
        self.assertIsNotNone(unsafe_reason("verus! { fn f() ensures true {} }"))


if __name__ == "__main__":
    unittest.main()
