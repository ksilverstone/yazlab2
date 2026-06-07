import numpy as np


class TimeSeriesSymbolizer:
    """Zaman serilerini ayrık sembollere (SAX) ve sıkıştırılmış formlara (PAA) dönüştüren sınıf."""

    def __init__(self, paa_size: int = 4, alphabet_size: int = 3):
        self.paa_size = paa_size
        self.alphabet_size = alphabet_size
        self.bins = None

    def _apply_paa(self, data_1d: np.ndarray) -> np.ndarray:
        """1 boyutlu diziyi paa_size büyüklüğündeki parçalara böler ve ortalamalarını alır."""
        n = len(data_1d)
        remainder = n % self.paa_size
        valid_n = n - remainder

        if valid_n == 0:
            return np.array([np.mean(data_1d)])

        reshaped_data = data_1d[:valid_n].reshape(-1, self.paa_size)
        paa_means = np.mean(reshaped_data, axis=1)

        if remainder > 0:
            last_mean = np.mean(data_1d[valid_n:])
            paa_means = np.append(paa_means, last_mean)

        return paa_means

    def fit(self, X_train_1d: np.ndarray):
        """Eğitim verisinden PAA uygulayarak SAX eşik değerlerini öğrenir."""
        paa_data = self._apply_paa(X_train_1d)
        percentiles = np.linspace(0, 100, self.alphabet_size + 1)[1:-1]
        self.bins = np.percentile(paa_data, percentiles)

    def transform(self, X_1d: np.ndarray) -> str:
        """Yeni gelen veriyi PAA ile sıkıştırıp daha önce öğrenilen sınırlara göre harflere çevirir."""
        if self.bins is None:
            raise ValueError("Transform işleminden önce fit() çağrılmalıdır!")

        paa_data = self._apply_paa(X_1d)
        indices = np.digitize(paa_data, self.bins)
        symbols = [chr(97 + i) for i in indices]
        return "".join(symbols)


class AutomataModel:
    """SAX sembol dizilerinden durum çıkarımı ve olasılıksal matris inşası için sınıf."""

    def __init__(self, window_size: int = 4):
        self.window_size = window_size
        self.transition_matrix = {}
        self.known_states = set()
        self.anomaly_threshold = 0.05

    def _extract_patterns(self, sax_string: str) -> list:
        """Uzun SAX karakter dizisini window_size uzunluğunda kayan pencerelere böler."""
        patterns = []
        n = len(sax_string)
        if n < self.window_size:
            return patterns
        for i in range(n - self.window_size + 1):
            pattern = sax_string[i : i + self.window_size]
            patterns.append(pattern)
        return patterns

    def fit(self, sax_string_train: str, laplace_alpha: float = 1e-5):
        """Eğitim verisinden Frekans Tabanlı Geçiş Olasılık Matrisi (TPM) oluşturur. Laplace Smoothing içerir."""
        patterns = self._extract_patterns(sax_string_train)
        self.known_states = set(patterns)
        counts = {}

        for i in range(len(patterns) - 1):
            curr_state = patterns[i]
            next_state = patterns[i + 1]

            if curr_state not in counts:
                counts[curr_state] = {}
            if next_state not in counts[curr_state]:
                counts[curr_state][next_state] = 0
            counts[curr_state][next_state] += 1

        self.num_observed_transitions = sum(len(t) for t in counts.values())

        n_states = len(self.known_states)
        for curr_state, transitions in counts.items():
            total_transitions = sum(transitions.values()) + (laplace_alpha * n_states)
            self.transition_matrix[curr_state] = {}
            for target_state in self.known_states:
                count = transitions.get(target_state, 0)
                prob = (count + laplace_alpha) / total_transitions
                self.transition_matrix[curr_state][target_state] = prob

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Dinamik Programlama ile iki string arasındaki Minimum Edit Distance'ı hesaplar."""
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
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        return dp[m][n]

    def _map_unseen_pattern(self, unseen_pattern: str) -> tuple:
        """Görülmemiş bir pattern'ı bilinen durumlara en yakın olanla eşler."""
        if not self.known_states:
            return None, float("inf")

        nearest_pattern = None
        min_distance = float("inf")

        for known_state in self.known_states:
            dist = self._levenshtein_distance(unseen_pattern, known_state)
            if dist < min_distance:
                min_distance = dist
                nearest_pattern = known_state

        return nearest_pattern, min_distance

    def get_step_probabilities(self, sax_string: str) -> np.ndarray:
        """Her zaman adımı için geçiş ve yol olasılıklarını hesaplar."""
        patterns = self._extract_patterns(sax_string)
        if not patterns:
            return np.array([])
            
        epsilon = 1e-5
        states = []
        for p in patterns:
            if p in self.known_states:
                states.append(p)
            else:
                mapped, _ = self._map_unseen_pattern(p)
                states.append(mapped)

        probs = []
        for i in range(len(patterns)):
            p1 = 1.0
            if i > 0:
                prev_state = states[i - 1]
                current_state = states[i]
                if prev_state in self.transition_matrix:
                    p1 = self.transition_matrix[prev_state].get(current_state, epsilon)
                else:
                    p1 = epsilon

            p2 = 1.0
            if i < len(patterns) - 1:
                current_state = states[i]
                next_state = states[i + 1]
                if current_state in self.transition_matrix:
                    p2 = self.transition_matrix[current_state].get(next_state, epsilon)
                else:
                    p2 = epsilon

            step_path_prob = p1 * p2
            probs.append(step_path_prob)
            
        return np.array(probs)

    def tune_threshold(self, val_sax_string: str, y_val: np.ndarray, paa_size: int) -> float:
        """Validation verisi üzerinde F1 skorunu maksimize edecek eşik değerini bulur."""
        probs = self.get_step_probabilities(val_sax_string)
        if len(probs) == 0:
            return 0.05

        best_threshold = 0.05
        best_f1 = -1.0
        
        # Arama uzayı: olasılıkların farklı yüzdelik dilimleri
        candidates = np.percentile(probs, np.linspace(0.1, 99.9, 100))
        
        y_val_arr = np.array(y_val)
        from sklearn.metrics import f1_score
        
        for threshold in candidates:
            preds = [1 if p < threshold else 0 for p in probs]
            
            aligned_preds = np.zeros(len(y_val_arr), dtype=int)
            for i, p in enumerate(preds):
                start_idx = (i + self.window_size - 1) * paa_size
                end_idx = min(start_idx + paa_size, len(y_val_arr))
                if p == 1:
                    aligned_preds[start_idx:end_idx] = 1
                    
            f1 = f1_score(y_val_arr, aligned_preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                
        self.anomaly_threshold = best_threshold
        return best_threshold

    def predict_and_explain(self, test_sax_string: str, anomaly_threshold: float = None) -> list:
        """
        Test verisi üzerinde adım adım geçiş olasılıklarını hesaplar
        ve açıklanabilir (explainable) JSON sonuçları üretir.
        """
        if anomaly_threshold is None:
            anomaly_threshold = self.anomaly_threshold

        patterns = self._extract_patterns(test_sax_string)
        results = []

        epsilon = 1e-5

        # 1. Tüm pattern'lar için durumları ve durum tiplerini belirle
        states = []
        statuses = []
        mapped_tos = []
        mapping_distances = []
        for p in patterns:
            if p in self.known_states:
                states.append(p)
                statuses.append("seen")
                mapped_tos.append(None)
                mapping_distances.append(0)
            else:
                mapped, dist = self._map_unseen_pattern(p)
                states.append(mapped)
                statuses.append("unseen")
                mapped_tos.append(mapped)
                mapping_distances.append(dist)

        # 2. Her zaman adımı için geçiş ve yol olasılıklarını hesapla
        for i in range(len(patterns)):
            current_state = states[i]
            raw_pattern = patterns[i]
            status = statuses[i]
            mapped_to = mapped_tos[i]
            mapping_distance = mapping_distances[i]

            p1 = 1.0
            p1_detail = None
            if i > 0:
                prev_state = states[i - 1]
                if prev_state in self.transition_matrix:
                    p1 = self.transition_matrix[prev_state].get(current_state, epsilon)
                else:
                    p1 = epsilon
                p1_detail = {"from": prev_state, "to": current_state, "probability": round(float(p1), 6)}

            p2 = 1.0
            p2_detail = None
            if i < len(patterns) - 1:
                next_state = states[i + 1]
                if current_state in self.transition_matrix:
                    p2 = self.transition_matrix[current_state].get(next_state, epsilon)
                else:
                    p2 = epsilon
                p2_detail = {"from": current_state, "to": next_state, "probability": round(float(p2), 6)}

            # Path probability is the product of entering and leaving transitions (covers 3 states)
            step_path_prob = p1 * p2

            transitions_detail = []
            if p1_detail is not None:
                transitions_detail.append(p1_detail)
            if p2_detail is not None:
                transitions_detail.append(p2_detail)

            # Confidence score is based on the transition step probability
            confidence_score = round(float(step_path_prob), 6)

            # Anomali kararı path olasılığına göre verilir
            decision = "anomaly" if step_path_prob < anomaly_threshold else "normal"

            step_result = {
                "time_step": i,
                "state": states[i - 1] if i > 0 else None,
                "pattern": raw_pattern,
                "status": status,
                "mapped_to": mapped_to,
                "mapping_distance": mapping_distance,
                "transitions": transitions_detail,
                "probability": round(float(step_path_prob), 6),
                "confidence_score": confidence_score,
                "decision": decision,
            }
            results.append(step_result)

        return results

    def predict_labels(self, test_sax_string: str, anomaly_threshold: float = None) -> list:
        """Test verisi için binary anomali etiketleri üretir (0: normal, 1: anomaly)."""
        if anomaly_threshold is None:
            anomaly_threshold = self.anomaly_threshold
        explanations = self.predict_and_explain(test_sax_string, anomaly_threshold)
        return [1 if e["decision"] == "anomaly" else 0 for e in explanations]

    def get_state_count(self) -> int:
        """Eğitimde öğrenilen benzersiz durum (state) sayısını döner."""
        return len(self.known_states)

    def get_transition_density(self) -> float:
        """Geçiş matrisinin yoğunluğunu (doluluk oranı) hesaplar."""
        if not self.known_states:
            return 0.0
        n_states = len(self.known_states)
        max_transitions = n_states * n_states
        actual_transitions = getattr(self, "num_observed_transitions", 0)
        return actual_transitions / max_transitions if max_transitions > 0 else 0.0
