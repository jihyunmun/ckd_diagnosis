import os
import numpy as np
import librosa
import random
from config import Config

def get_max_duration_in_dir(audio_dir, sr=16000):
    max_duration = 0.0
    for fname in os.listdir(audio_dir):
        if fname.endswith('.wav'):
            path = os.path.join(audio_dir, fname)
            y, _ = librosa.load(path, sr=sr, mono=True)
            duration = len(y) / sr
            if duration > max_duration:
                max_duration = duration
    return max_duration

class GetMelSpectrogram:
    def __init__(self, n_mels=256, n_fft=2048, hop_length=512, sr=16000, fmin=0, fmax=None, window='hann'):
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sr = sr
        self.fmin = fmin
        self.fmax = fmax
        self.window = window

    def pad_or_truncate_audio(self, y, target_duration):
        target_length = int(target_duration * self.sr)
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), 'constant')
        elif len(y) > target_length:
            y = y[:target_length]

        return y
    
    def create_spectrogram(self, path, target_duration):
        y, _ = librosa.load(path, sr=self.sr, mono=True)
        y = self.pad_or_truncate_audio(y, target_duration)
        S = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop_length, fmin=self.fmin, fmax=self.fmax, window=self.window)
        S_dB = librosa.power_to_db(S, ref=np.max)
        return S_dB
    
def run_spectrogram_extraction():
    splits = [
        ('train', Config.TRAIN_DIR, Config.SPEC_TRAIN_DIR),
        ('val', Config.VAL_DIR, Config.SPEC_VAL_DIR),
        ('test', Config.TEST_DIR, Config.SPEC_TEST_DIR)
    ]

    global_max_duration = 0.0
    for split_name, audio_dir, _ in splits:
        if not os.path.exists(audio_dir):
            print(f"[WARNING] {split_name} directory does not exist: {audio_dir}")
            continue
        max_dur = get_max_duration_in_dir(audio_dir, sr=16000)
        if max_dur > global_max_duration:
            global_max_duration = max_dur

    mel_extractor = GetMelSpectrogram(sr=16000)

    for split_name, audio_dir, spec_dir in splits:
        if not os.path.exists(audio_dir):
            print(f"[WARNING] {split_name} directory does not exist: {audio_dir}")
            continue
        os.makedirs(spec_dir, exist_ok=True)

        audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
        for wav_name in audio_files:
            wav_path = os.path.join(audio_dir, wav_name)
            spec_path = os.path.join(spec_dir, wav_name.replace('.wav', '.npy'))

            melspectrogram = mel_extractor.create_spectrogram(wav_path, global_max_duration)
            np.save(spec_path, melspectrogram)

        print(f"[INFO] Generated spectrograms for {split_name} set")

    print(f"[INFO] Spectrogram extraction completed")
