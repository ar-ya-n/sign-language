import unittest
from signvision.history import RecognitionHistory
from signvision.stabilizer import PredictionStabilizer
from signvision.word_builder import WordBuilder


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

    def test_word_builder_rearms_repeated_letters(self):
        builder = WordBuilder()
        builder.update("B")
        builder.update("B")
        self.assertEqual("B", builder.word)
        builder.update(None)
        builder.update("O")
        builder.update(None)
        builder.update("O")
        self.assertEqual("BOO", builder.word)


if __name__ == "__main__":
    unittest.main()
