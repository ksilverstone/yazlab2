import unittest
from src.models.automata import AutomataModel

class TestLevenshteinMapping(unittest.TestCase):
    
    def setUp(self):
        """Her testten önce çalıştırılan hazırlık aşaması."""
        # 3 harfli pencerelerle çalışan bir test objesi oluşturuyoruz
        self.automata = AutomataModel(window_size=3)
        
        # Sahte (mock) bilinen SAX pattern'larını (state) manuel olarak ekliyoruz
        self.automata.known_states = {"aab", "abc", "bcc"}

    def test_levenshtein_distance(self):
        """_levenshtein_distance metodunun doğru hesaplama yapıp yapmadığını test eder."""
        # Klasik Edit Distance örneği: kitten -> sitting (Mesafe: 3)
        dist1 = self.automata._levenshtein_distance("kitten", "sitting")
        self.assertEqual(dist1, 3, "Kitten ve sitting arasındaki mesafe 3 olmalıdır.")
        
        # Proje özelindeki SAX karakterleriyle test: aab -> abc (Mesafe: 2)
        dist2 = self.automata._levenshtein_distance("aab", "abc")
        self.assertEqual(dist2, 2, "aab ve abc arasındaki mesafe 2 olmalıdır.")

    def test_map_unseen_pattern(self):
        """Görülmemiş (unseen) bir pattern'ın sözlükteki en yakın harf dizilimine doğru eşlenmesini test eder."""
        unseen_pattern = "aac"
        
        # "aac" ile bilinen durumlar arasındaki mesafeler:
        # "aac" -> "aab" (Mesafe: 1)
        # "aac" -> "abc" (Mesafe: 2)
        # "aac" -> "bcc" (Mesafe: 3)
        # Beklenen çıktı: ("aab", 1)
        
        nearest_pattern, min_distance = self.automata._map_unseen_pattern(unseen_pattern)
        
        self.assertEqual(min_distance, 1, "Minimum mesafe 1 olarak hesaplanmalıdır.")
        self.assertEqual(nearest_pattern, "aab", "En yakın pattern 'aab' olarak eşlenmelidir.")

if __name__ == '__main__':
    unittest.main()
