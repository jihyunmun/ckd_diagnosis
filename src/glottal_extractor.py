import os
import warnings
import pandas as pd
from numpy import VisibleDeprecationWarning
from disvoice.glottal import Glottal

from config import Config

def extract_glottal_features_from_folder(audio_dir, output_csv):
    glottalf = Glottal()
    file_names = os.listdir(audio_dir)
    file_names.sort()
    total_df = pd.DataFrame()

    for name in file_names:
        if not name.endswith('.wav'):
            continue
        wav_path = os.path.join(audio_dir, name)
        try:
            features = glottalf.extract_features_file(wav_path, static=True, plots=False, fmt='dataframe')
            features['filename'] = name.replace('.wav', '')
            total_df = pd.concat([total_df, features], ignore_index=True)
        except (RuntimeWarning, VisibleDeprecationWarning) as e:
            print(f"Error extracting glottal features for {name}: {e}")

    total_df.to_csv(output_csv, index=False)
    print(f"[INFO] Glottal features saved to {output_csv}")

def run_glottal_extraction():
    if os.path.exists(Config.TRAIN_DIR):
        extract_glottal_features_from_folder(Config.TRAIN_DIR, Config.TRAIN_GLOTTAL_CSV)
    else:
        print(f"[WARNING] Train dir not found: {Config.TRAIN_DIR}")
    
    if os.path.exists(Config.VALID_DIR):
        extract_glottal_features_from_folder(Config.VALID_DIR, Config.VALID_GLOTTAL_CSV)
    else:
        print(f"[WARNING] Valid dir not found: {Config.VALID_DIR}")

    if os.path.exists(Config.TEST_DIR):
        extract_glottal_features_from_folder(Config.TEST_DIR, Config.TEST_GLOTTAL_CSV)
    else:
        print(f"[WARNING] Test dir not found: {Config.TEST_DIR}")