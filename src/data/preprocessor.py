import pandas as pd
import yaml
from pathlib import Path

class SKABPreprocessor:
    """SKAB veri setini okuma, birleştirme ve X, y, groups ayrımını yapma sınıfı."""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        self.raw_dir = Path(self.config['data']['skab']['raw_dir'])
        self.group_col = self.config['data']['skab']['group_col']
        self.target_col = 'anomaly'
        self.exclude_cols = ['datetime', 'changepoint', 'source_group', 'source_file', self.target_col]
        
    def load_and_merge_data(self):
        dataframes = []
        target_folders = ['valve1', 'valve2']
        
        for folder_name in target_folders:
            folder_path = self.raw_dir / folder_name
            
            if not folder_path.exists():
                continue
                
            for file_path in folder_path.glob('*.csv'):
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
        
        y = merged_df[self.target_col] if self.target_col in merged_df.columns else pd.Series(dtype=int)
        groups = merged_df[self.group_col] if self.group_col in merged_df.columns else pd.Series(dtype=str)
        
        cols_to_drop = [col for col in self.exclude_cols if col in merged_df.columns]
        X = merged_df.drop(columns=cols_to_drop)
        
        return X, y, groups
