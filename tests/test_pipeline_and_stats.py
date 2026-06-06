import unittest
from scipy.stats import wilcoxon

class TestPipelineStats(unittest.TestCase):
    def test_wilcoxon_test(self):
        """İki modelin F1 skorları arasındaki anlamlılık testinin çökmediğini doğrular."""
        # Sahte skor listeleri
        m1_scores = [0.85, 0.86, 0.84, 0.88, 0.89]
        m2_scores = [0.75, 0.74, 0.78, 0.76, 0.72]
        
        w_stat, p_val = wilcoxon(m1_scores, m2_scores, zero_method="zsplit")
        # Wilcoxon sonucu p-value döndürmelidir.
        self.assertIsNotNone(p_val)

    def test_wilcoxon_identical_scores(self):
        """Aynı skorlar verildiğinde de çökme olmamalıdır (Zero difference durumu)."""
        m1_scores = [0.85, 0.85, 0.85]
        m2_scores = [0.85, 0.85, 0.85]
        w_stat, p_val = wilcoxon(m1_scores, m2_scores, zero_method="zsplit")
        self.assertIsNotNone(p_val)

if __name__ == '__main__':
    unittest.main()
