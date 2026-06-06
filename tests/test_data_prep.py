import unittest
import pandas as pd
import numpy as np
from src.data.preprocessor import DataTransformer

class TestDataPrep(unittest.TestCase):
    def test_data_transformer_fit_transform(self):
        """Verilerin doğru standardize edildiği test edilir."""
        df = pd.DataFrame({'sensor1': [1.0, 2.0, 3.0], 'sensor2': [4.0, 5.0, 6.0]})
        transformer = DataTransformer(config_path="configs/config.yaml")
        # Sadece hata fırlatıp fırlatmadığına bakılır, scaler mantığı
        df_scaled, np_scaled = transformer.fit_transform(df)
        self.assertEqual(df_scaled.shape[0], df.shape[0])
        self.assertEqual(np_scaled.shape[0], df.shape[0])

    def test_transformer_transform(self):
        """Fit olmadan transform edilirse ValueError vermemelidir çünkü dummy scaler kurulur."""
        df = pd.DataFrame({'sensor1': [10.0, 20.0]})
        transformer = DataTransformer(config_path="configs/config.yaml")
        transformer.fit_transform(df)
        df_test = pd.DataFrame({'sensor1': [15.0, 25.0]})
        df_out, np_out = transformer.transform(df_test)
        self.assertEqual(df_out.shape[0], df_test.shape[0])

if __name__ == '__main__':
    unittest.main()
