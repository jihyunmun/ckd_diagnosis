import os
import argparse

from config import Config
from spectrogram_generator import run_spectrogram_extraction
from glottal_extractor import run_glottal_extraction
from train_loop import run_train, run_inference
from ensemble_analysis import run_analysis

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train',
                        help='Options: spectrogram, glottal, train, inference, analysis, all')
    return parser.parse_args()

def main():
    args = parse_args()
    mode = args.mode.lower()

    if mode == 'spectrogram':
        run_spectrogram_extraction()
        return

    elif mode == 'glottal':
        run_glottal_extraction()
        return

    elif mode == 'train':
        run_train()
        return
    
    elif mode == 'inference':
        run_inference()
        return
    
    elif mode == 'analysis':
        run_analysis()
        return
    
    elif mode == 'all':
        run_spectrogram_extraction()
        run_glottal_extraction()
        run_train()
        run_inference()
        run_analysis()
        return
    
if __name__ == '__main__':
    main()