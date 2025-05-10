from torch.utils.data import Dataset
import h5py
import numpy as np

class MarsDataset(Dataset):

    def __init__(self, file_path):

        with h5py.File(file_path, 'r') as f:
            
            self.cell_repr = np.asarray(f['cell_repr'])
            self.smile_repr = np.asarray(f['smile_repr'])
            self.dosage = np.asarray(f['dosage'])
            self.vision_scores = np.asarray(f['vision_scores'])

    def __len__(self):

        return len(self.cell_repr)
    
    def __getitem__(self, idx):

        cell_repr = self.cell_repr[idx]
        smile_repr = self.smile_repr[idx]
        dosage = self.dosage[idx]
        vision_scores = self.vision_scores[idx]

        return cell_repr, smile_repr, dosage, vision_scores