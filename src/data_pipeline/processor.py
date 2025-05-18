import pandas as pd
from sklearn.model_selection import train_test_split

class ProteinDataProcessor:
    def prepare_data(self, df, test_size=0.2, random_state=42):
        # Drop rows with missing label
        df = df.dropna(subset=['ptm_label'])
        # Select features and label
        feature_cols = [
            'sequence_length', 'molecular_weight', 'n_cysteines',
            'n_domains', 'n_disulfide_bonds', 'has_signal_peptide',
            'has_transmem', 'taxonomy_id'
        ]
        X = df[feature_cols].fillna(0)
        y = df['ptm_label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        return X_train, X_test, y_train, y_test 