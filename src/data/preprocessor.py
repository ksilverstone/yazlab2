import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class SKABPreprocessor:
    """SKAB veri setini okuma, birleştirme ve X, y, groups ayrımını yapma sınıfı."""

    def __init__(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.raw_dir = Path(self.config['data']['skab']['raw_dir'])
        self.group_col = self.config['data']['skab'].get('group_col', 'source_file')
        self.target_col = self.config['data']['skab'].get('target_col', 'anomaly')
        self.exclude_cols = ['datetime', 'changepoint', 'source_group', 'source_file', self.target_col]

    def load_and_merge_data(self):
        """valve1 ve valve2 klasörlerindeki CSV dosyalarını birleştirip X, y, groups olarak ayırır."""
        dataframes = []
        target_folders = self.config['data']['skab'].get('folders', ['valve1', 'valve2'])

        for folder_name in target_folders:
            folder_path = self.raw_dir / folder_name

            if not folder_path.exists():
                continue

            for file_path in sorted(folder_path.glob('*.csv')):
                try:
                    df = pd.read_csv(file_path, sep=';')
                    if len(df.columns) == 1:
                        df = pd.read_csv(file_path, sep=',')

                    df['source_group'] = folder_name
                    df['source_file'] = file_path.name
                    dataframes.append(df)
                except Exception:
                    pass

        if not dataframes:
            return pd.DataFrame(), pd.Series(dtype=int), pd.Series(dtype=str)

        merged_df = pd.concat(dataframes, ignore_index=True)

        y = merged_df[self.target_col].astype(int) if self.target_col in merged_df.columns else pd.Series(dtype=int)
        groups = merged_df[self.group_col] if self.group_col in merged_df.columns else pd.Series(dtype=str)

        cols_to_drop = [col for col in self.exclude_cols if col in merged_df.columns]
        X = merged_df.drop(columns=cols_to_drop)

        # Numerik olmayan sütunları düşür
        X = X.select_dtypes(include=[np.number])

        return X, y, groups

    def get_group_kfold_splits(self, X, y, groups, n_splits=5):
        """GroupKFold ile dosya bazlı bölme — aynı dosya hem train hem test'te olmaz."""
        gkf = GroupKFold(n_splits=n_splits)
        return list(gkf.split(X, y, groups))


class BATADALPreprocessor:
    """BATADAL veri setini okuma, temizleme ve kronolojik olarak Train/Val/Test ayırma sınıfı."""

    def __init__(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.file_path = Path(self.config['data']['batadal']['file_path'])
        self.train_ratio = self.config['data']['batadal']['train_ratio']
        self.val_ratio = self.config['data']['batadal']['val_ratio']
        self.target_col = self.config['data']['batadal'].get('target_col', 'ATT_FLAG')
        self.exclude_cols = ['DATETIME', 'datetime', self.target_col]

    def load_and_split_data(self):
        """Veriyi okur ve sırasını bozmadan %60, %20, %20 böler."""
        if not self.file_path.exists():
            empty_x = pd.DataFrame()
            empty_y = pd.Series(dtype=int)
            return empty_x, empty_y, empty_x, empty_y, empty_x, empty_y

        df = pd.read_csv(self.file_path)

        # Sütun isimlerindeki boşlukları temizle
        df.columns = df.columns.str.strip()

        # ATT_FLAG: -999 -> 0 (normal), 1 -> 1 (anomali)
        if self.target_col in df.columns:
            df[self.target_col] = df[self.target_col].apply(lambda x: 0 if x == -999 else 1)
            y = df[self.target_col].astype(int)
        else:
            y = pd.Series(dtype=int)

        cols_to_drop = [col for col in self.exclude_cols if col in df.columns]
        X = df.drop(columns=cols_to_drop)

        # Numerik olmayan sütunları düşür
        X = X.select_dtypes(include=[np.number])

        # Eksik değerleri ileri doldurma (forward fill) ile tamamla
        X = X.ffill().bfill().fillna(0)

        # Zamansal (Chronological) Bölme
        n = len(df)
        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.val_ratio)

        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
        X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

        return X_train, y_train, X_val, y_val, X_test, y_test


class DataTransformer:
    """Data Leakage korumalı, StandardScaler ve PCA uygulayan dönüşüm sınıfı."""

    def __init__(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.config['preprocessing']['pca_n_components'])
        self._fitted = False

    def fit_transform(self, X_train):
        """Yalnızca Train verisine uygulanmalı. (DL verisi, Automata verisi) tuple döner."""
        X_dl = self.scaler.fit_transform(X_train)
        X_automata = self.pca.fit_transform(X_dl)
        self._fitted = True
        return X_dl, X_automata

    def transform(self, X_val_or_test):
        """Train verisiyle öğrenilen scaler ve pca ile dönüştürür."""
        if not self._fitted:
            raise ValueError("fit_transform() önce çağrılmalıdır!")
        X_dl = self.scaler.transform(X_val_or_test)
        X_automata = self.pca.transform(X_dl)
        return X_dl, X_automata
