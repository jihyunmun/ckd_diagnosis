import torch
import random
import numpy as np

class Config:
    # prepare the data
    TRAIN_DIR = 'data/audio_split/train'
    VALID_DIR = 'data/audio_split/valid'
    TEST_DIR = 'data/audio_split/test'

    SPEC_TRAIN_DIR = 'data/spectrogram/train'
    SPEC_VALID_DIR = 'data/spectrogram/valid'
    SPEC_TEST_DIR = 'data/spectrogram/test'

    TRAIN_GLOTTAL_CSV = 'train_glottal.csv'
    VALID_GLOTTAL_CSV = 'valid_glottal.csv'
    TEST_GLOTTAL_CSV = 'test_glottal.csv'

    CHECKPOINT_DIR = 'checkpoints'
    OUTPUT_JSON = 'output.json'

    # Training settings
    SEEDS = [0, 1, 2, 3, 4]
    BATCH_SIZE = 16
    NUM_EPOCHS = 200
    PATIENCE = 20
    LR = 1e-4
    WEIGHT_DECAY = 1e-2
    NUM_WORKERS = 16

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed: int):
    """
    Set seed for reproducibility
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False