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

    def predict_and_explain(self, test_sax_string: str, anomaly_threshold: float = 0.05) -> list:
        """
        Test verisi üzerinde adım adım geçiş olasılıklarını hesaplar
        ve açıklanabilir (explainable) JSON sonuçları üretir.

        Çıktı formatı PDF gereksinimlerini karşılar:
        - state, pattern, status, mapped_to
        - transitions listesi
        - probability (geçiş olasılığı)
        - path_probability (kümülatif)
        - confidence_score
        - decision
        """
        patterns = self._extract_patterns(test_sax_string)
        results = []

        epsilon = 1e-5
        cumulative_log_prob = 0.0

        for i in range(len(patterns)):
            raw_pattern = patterns[i]

            # 1. Durum Kontrolü ve Haritalama
            if raw_pattern in self.known_states:
                status = "seen"
                mapped_to = None
                mapping_distance = 0
                current_state = raw_pattern
            else:
                status = "unseen"
                mapped_to, mapping_distance = self._map_unseen_pattern(raw_pattern)
                current_state = mapped_to

            # 2. Geçiş Olasılığı Hesabı
            transition_prob = epsilon
            transitions_detail = []

            if i < len(patterns) - 1:
                next_raw = patterns[i + 1]

                if next_raw in self.known_states:
                    next_state = next_raw
                else:
                    next_state = self._map_unseen_pattern(next_raw)[0]

                # Mevcut state'in tüm geçişlerini listele (yalnızca ihtimali yüksek olanlar)
                if current_state in self.transition_matrix:
                    for target, prob in self.transition_matrix[current_state].items():
                        if (
                            prob > 1e-4
                        ):  # Çok düşük (sadece laplace'dan gelen) olasılıkları filtrele
                            transitions_detail.append(
                                {"from": current_state, "to": target, "probability": round(prob, 6)}
                            )
                    transition_prob = self.transition_matrix[current_state].get(next_state, epsilon)
            else:
                transition_prob = 1.0

            # 3. Path Probability (kümülatif, log-space underflow korumalı)
            cumulative_log_prob += np.log(max(transition_prob, epsilon))
            path_probability = float(np.exp(cumulative_log_prob))

            # 4. Güven Skoru (Confidence Score)
            # Yüksek geçiş olasılığı = yüksek güven
            confidence_score = round(float(transition_prob), 6)

            # 5. Anomali Kararı
            decision = "anomaly" if transition_prob < anomaly_threshold else "normal"

            step_result = {
                "time_step": i,
                "state": current_state,
                "pattern": raw_pattern,
                "status": status,
                "mapped_to": mapped_to,
                "mapping_distance": mapping_distance,
                "transitions": transitions_detail,
                "probability": round(float(transition_prob), 6),
                "path_probability": round(path_probability, 10),
                "confidence_score": confidence_score,
                "decision": decision,
            }
            results.append(step_result)

        return results

    def predict_labels(self, test_sax_string: str, anomaly_threshold: float = 0.05) -> list:
        """Test verisi için binary anomali etiketleri üretir (0: normal, 1: anomaly)."""
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
        actual_transitions = sum(len(v) for v in self.transition_matrix.values())
        return actual_transitions / max_transitions if max_transitions > 0 else 0.0
