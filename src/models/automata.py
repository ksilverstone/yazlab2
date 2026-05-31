import numpy as np

class TimeSeriesSymbolizer:
    """Zaman serilerini ayrık sembollere (SAX) ve sıkıştırılmış formlara (PAA) dönüştüren sınıf."""
    
    def __init__(self, paa_size: int = 4, alphabet_size: int = 3):
        self.paa_size = paa_size
        self.alphabet_size = alphabet_size
        self.bins = None  # SAX için daha sonra hesaplanacak sınır değerleri
        
    def _apply_paa(self, data_1d: np.ndarray) -> np.ndarray:
        """1 boyutlu diziyi paa_size büyüklüğündeki parçalara böler ve ortalamalarını alır."""
        n = len(data_1d)
        remainder = n % self.paa_size
        valid_n = n - remainder
        
        # Eğer dizi paa_size'dan kısa ise tüm dizinin ortalamasını al ve dön
        if valid_n == 0:
            return np.array([np.mean(data_1d)])
            
        # Tam bölünebilen kısmı yeniden şekillendirip satır bazlı ortalama (mean) alıyoruz
        reshaped_data = data_1d[:valid_n].reshape(-1, self.paa_size)
        paa_means = np.mean(reshaped_data, axis=1)
        
        # Tam bölünmeyen artan/sondaki bir parça varsa, onun da ortalamasını sona ekliyoruz
        if remainder > 0:
            last_mean = np.mean(data_1d[valid_n:])
            paa_means = np.append(paa_means, last_mean)
            
        return paa_means
