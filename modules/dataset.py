from torch.utils.data import Dataset
import h5py
import numpy as np

class MarsDataset(Dataset):

    def __init__(self, file_path):

        with h5py.File(file_path, 'r') as f:
            
            # self.cell_id = np.asarray(f['cell_ids'])
            # self.cell_name = np.asarray(f['cell_names'])
            self.cell_repr = np.asarray(f['X_uce'])
            # self.drug_name = np.asarray(f['drug'])
            self.smile_repr = np.asarray(f['drug_embeddings'])
            self.dosage = np.asarray(f['drugname_drugconc'])
            self.vision_scores = np.asarray(f['vision_scores'])

    def __len__(self):

        return len(self.cell_repr)
    
    def __getitem__(self, idx):

        # cell_id = self.cell_id[idx]
        # cell_name = self.cell_name[idx]
        cell_repr = self.cell_repr[idx]
        # drug_name = self.drug_name[idx]
        smile_repr = self.smile_repr[idx]
        dosage = self.dosage[idx]
        vision_scores = self.vision_scores[idx]

        # return (cell_id, cell_name, cell_repr, drug_name, smile_repr, dosage), vision_scores
        return (cell_repr, smile_repr, dosage), vision_scores