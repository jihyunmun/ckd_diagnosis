import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

###########################################################
# Spectrogram only Dataset
###########################################################
class SpectrogramDataset(Dataset):
    def __init__(self,
                 spec_dir,
                 resize_to=(224, 224)):
        """
        load spectrogram(npz) files
        """
        self.data = []
        if os.path.exists(spec_dir):
            spec_files = [os.path.join(spec_dir, f) for f in os.listdir(spec_dir) if f.endswith('.npy')]
            self.data.extend(spec_files)
        else:
            print(f"[WARNING] Spectrogram dir not found: {spec_dir}")
            spec_files = []
        self.data = sorted(self.data)
        self.resize_to = resize_to

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        path = self.data[idx]
        data = np.load(path)

        if data.ndim == 2:
            data = data[np.newaxis, ...] # (1, freq, time)
        data = np.repeat(data, 3, axis=0) # (3, freq, time) -> to fit the input channel size of ResNet

        if self.resize_to is not None:
            data = F.interpolate(data.unsqueeze(0), size=self.resize_to, mode='bilinear', align_corners=False)
            data = data.squeeze(0)
        
        data = torch.tensor(data, dtype=torch.float32)
        labels = self._get_label_from_path(path)
        return data, labels
    
    def _get_label_from_path(self, path):
        """
        (gender)_(label)_(patient number).npy -> extract label from path
        """
        filename = os.path.basename(path)
        label_str = filename.split('_')[1]

        if label_str == 'N': # non-CKD
            diag_label = 0
            eval_label = 0
        else:
            diag_label = 1
            eval_label = int(label_str)

        return {
            'diag': torch.tensor(diag_label, dtype=torch.float32),
            'eval': torch.tensor(eval_label, dtype=torch.float32)
        }
    
###########################################################
# Spectrogram + Glottal feature Dataset
###########################################################
class SpectrogramGlottalDataset(SpectrogramDataset):
    def __init__(self, 
                 spec_dir,
                 glottal_csv,
                 normalize_glottal=False,
                 resize_to=(224,224)):
        super().__init__(spec_dir, resize_to=resize_to)

        self.glottal_features = [
            'global avg var GCI',
            'global avg avg NAQ',
            'global avg std NAQ',
            'global avg avg QOQ',
            'global avg std QOQ',
            'global avg avg H1H2',
            'global avg std H1H2',
            'global avg avg HRF',
            'global avg std HRF'
        ]

        self.glottal_df = pd.read_csv(glottal_csv)

        if normalize_glottal:
            self._normalize_glottal_df()

        self.glottal_dict = self._create_glottal_dict()

        # filter data in case glottal features are not available
        filtered_data = []
        for path in self.data:
            filename = os.path.splitext(os.path.basename(path))[0]
            if filename in self.glottal_dict:
                filtered_data.append(path)
        self.data = filtered_data

    def _normalize_glottal_df(self):
        mean_vals = self.glottal_df[self.glottal_features].mean()
        std_vals = self.glottal_df[self.glottal_features].std()
        self.glottal_df[self.glottal_features] = (self.glottal_df[self.glottal_features] - mean_vals) / std_vals

    def _create_glottal_dict(self):
        glottal_dict = {}
        for _, row in self.glottal_df.iterrows():
            filename = str(row['filename'])
            features = [row[feat] for feat in self.glottal_features]
            glottal_dict[filename] = torch.tensor(features, dtype=torch.float32)
        return glottal_dict
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data, labels = super().__getitem__(idx)
        path = self.data[idx]
        filename = os.path.splitext(os.path.basename(path))[0]
        glottal_features = self.glottal_dict[filename]
        return data, glottal_features, labels
