import unittest
import numpy as np
from src.models.automata import AutomataModel, TimeSeriesSymbolizer

class TestLevenshteinDistance(unittest.TestCase):
    def setUp(self):
        self.automata = AutomataModel(window_size=3)
        self.automata.known_states = {"aab", "abc", "bcc"}

    def test_identical_strings(self):
        self.assertEqual(self.automata._levenshtein_distance("abc", "abc"), 0)

    def test_empty_strings(self):
        self.assertEqual(self.automata._levenshtein_distance("", "abc"), 3)

    def test_sax_patterns(self):
        self.assertEqual(self.automata._levenshtein_distance("aab", "abc"), 2)

    def test_single_char_difference(self):
        self.assertEqual(self.automata._levenshtein_distance("aab", "aac"), 1)

class TestUnseenMapping(unittest.TestCase):
    def setUp(self):
        self.automata = AutomataModel(window_size=3)
        self.automata.known_states = {"aab", "abc", "bcc"}

    def test_map_nearest(self):
        nearest, distance = self.automata._map_unseen_pattern("aac")
        self.assertIn(nearest, ["aab", "abc"])
        self.assertEqual(distance, 1)

class TestSAXTransform(unittest.TestCase):
    def test_fit_creates_bins(self):
        symbolizer = TimeSeriesSymbolizer(paa_size=1, alphabet_size=3)
        symbolizer.fit(np.array([1.0, 2.0, 3.0, 4.0]))
        self.assertEqual(len(symbolizer.bins), 2)

    def test_transform_returns_string(self):
        symbolizer = TimeSeriesSymbolizer(paa_size=1, alphabet_size=3)
        data = np.array([1.0, 2.0, 3.0])
        symbolizer.fit(data)
        self.assertIsInstance(symbolizer.transform(data), str)

if __name__ == '__main__':
    unittest.main()
