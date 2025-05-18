import os
import json
import pandas as pd

class ProteinDataCollector:
    def __init__(self, protein_details_path=None):
        if protein_details_path is None:
            protein_details_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'assets', 'protein_details.json')
        with open(protein_details_path, 'r') as f:
            self.protein_data = json.load(f)

    def collect_ptm_data(self, ptm_type='CARBOHYD'):
        rows = []
        for entry in self.protein_data:
            details = entry['Details']
            sequence = details.get('sequence', {}).get('sequence', '')
            features = details.get('features', [])
            # PTM label: 1 if at least one feature of ptm_type exists
            ptm_label = int(any(f.get('type') == ptm_type for f in features))
            # Example features
            n_domains = sum(1 for f in features if f.get('type') == 'DOMAIN')
            n_disulfide = sum(1 for f in features if f.get('type') == 'DISULFID')
            has_signal = int(any(f.get('type') == 'SIGNAL' for f in features))
            has_transmem = int(any(f.get('type') == 'TRANSMEM' for f in features))
            n_cysteines = sequence.count('C')
            seq_len = len(sequence)
            mol_weight = details.get('sequence', {}).get('mass', None)
            protein_existence = details.get('proteinExistence', None)
            taxonomy_id = details.get('organism', {}).get('taxonomy', None)
            rows.append({
                'accession': entry['Accession'],
                'sequence_length': seq_len,
                'molecular_weight': mol_weight,
                'n_cysteines': n_cysteines,
                'n_domains': n_domains,
                'n_disulfide_bonds': n_disulfide,
                'has_signal_peptide': has_signal,
                'has_transmem': has_transmem,
                'protein_existence': protein_existence,
                'taxonomy_id': taxonomy_id,
                'ptm_label': ptm_label
            })
        return pd.DataFrame(rows) 