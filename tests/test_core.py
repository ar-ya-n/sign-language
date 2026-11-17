import unittest
from signvision.history import RecognitionHistory
from signvision.stabilizer import PredictionStabilizer


class CoreTests(unittest.TestCase):
    def test_stabilizer(self):
        s = PredictionStabilizer(3, 2 / 3)
        self.assertIsNone(s.update("A"))
        self.assertEqual("A", s.update("A"))
        self.assertEqual("A", s.update("B"))

    def test_history_limit(self):
        h = RecognitionHistory(2)
        h.add("A", 1); h.add("B", 1); h.add("C", 1)
        self.assertEqual(["B", "C"], [x.label for x in h.entries()])


if __name__ == "__main__":
    unittest.main()
