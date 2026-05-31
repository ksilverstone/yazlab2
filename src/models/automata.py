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

    def fit(self, X_train_1d: np.ndarray):
        """Eğitim verisinden PAA uygulayarak SAX eşik değerlerini öğrenir."""
        paa_data = self._apply_paa(X_train_1d)
        
        # alphabet_size kadar dilim için alphabet_size-1 adet eşik değeri gerekir
        # np.linspace ile yüzdelik (percentile) eşiklerini hesaplıyoruz (ör: %33.3, %66.6)
        percentiles = np.linspace(0, 100, self.alphabet_size + 1)[1:-1]
        self.bins = np.percentile(paa_data, percentiles)
        
    def transform(self, X_1d: np.ndarray) -> str:
        """Yeni gelen veriyi PAA ile sıkıştırıp daha önce öğrenilen sınırlara göre harflere çevirir."""
        if self.bins is None:
            raise ValueError("Transform işleminden önce fit() çağrılmalıdır!") # User doesn't want standard error prints directly, but raising exceptions for coding errors is fine.
            
        paa_data = self._apply_paa(X_1d)
        
        # numpy.digitize değerlerin hangi bin'e düştüğünün indeksini döner (0, 1, 2...)
        indices = np.digitize(paa_data, self.bins)
        
        # İndeksleri ASCII karakterlerine çevir: 0 -> 'a' (ASCII 97), 1 -> 'b', vs.
        symbols = [chr(97 + i) for i in indices]
        
        return "".join(symbols)


class AutomataModel:
    """SAX sembol dizilerinden durum (state) çıkarımı ve olasılıksal matris inşası için sınıf."""
    
    def __init__(self, window_size: int = 4):
        self.window_size = window_size
        self.transition_matrix = {}
        self.known_states = set()
        
    def _extract_patterns(self, sax_string: str) -> list:
        """Uzun SAX karakter dizisini window_size uzunluğunda kayan pencerelere (pattern) böler."""
        patterns = []
        n = len(sax_string)
        
        # window_size'dan daha kısa bir string gelirse hata oluşmasını engeller
        if n < self.window_size:
            return patterns
            
        # Kayan pencere (Sliding Window) mantığı ile adım adım (step=1) ilerleyerek pattern oluştur
        for i in range(n - self.window_size + 1):
            pattern = sax_string[i : i + self.window_size]
            patterns.append(pattern)
            
        return patterns

    def fit(self, sax_string_train: str):
        """Eğitim verisinden Frekans Tabanlı Geçiş Olasılık Matrisi (TPM) oluşturur."""
        patterns = self._extract_patterns(sax_string_train)
        counts = {}
        
        # Ardışık pattern'lar arası geçişleri (transition) say
        for i in range(len(patterns) - 1):
            curr_state = patterns[i]
            next_state = patterns[i + 1]
            
            if curr_state not in counts:
                counts[curr_state] = {}
            if next_state not in counts[curr_state]:
                counts[curr_state][next_state] = 0
                
            counts[curr_state][next_state] += 1
            
        # Olasılıkları hesapla ve transition_matrix'e kaydet
        for curr_state, transitions in counts.items():
            total_transitions = sum(transitions.values())
            self.transition_matrix[curr_state] = {}
            
            for next_state, count in transitions.items():
                prob = count / total_transitions
                self.transition_matrix[curr_state][next_state] = prob
                
        # Eğitilen tüm durumları kaydet (Unseen pattern eşlemesi için gerekli)
        self.known_states = set(patterns)

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Dinamik Programlama (DP) ile iki string arasındaki Minimum Edit Distance'ı hesaplar."""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0:
                    dp[i][j] = j
                elif j == 0:
                    dp[i][j] = i
                elif s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j],      # Silme
                                       dp[i][j - 1],      # Ekleme
                                       dp[i - 1][j - 1])  # Değiştirme
        return dp[m][n]

    def _map_unseen_pattern(self, unseen_pattern: str) -> tuple:
        """Görülmemiş bir pattern'ı bilinen durumlara en yakın (min Levenshtein mesafeli) olanla eşler."""
        if not self.known_states:
            return None, float('inf')
            
        nearest_pattern = None
        min_distance = float('inf')
        
        for known_state in sorted(self.known_states):
            dist = self._levenshtein_distance(unseen_pattern, known_state)
            if dist < min_distance:
                min_distance = dist
                nearest_pattern = known_state
                
        return nearest_pattern, min_distance
