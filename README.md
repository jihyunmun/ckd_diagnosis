# README.md

This repository provides a framework to diagnose chronic kidney disease.
The original paper is submitted to Interspeech 2025.

![Image](https://github.com/user-attachments/assets/01ff4098-3116-4d03-8502-b3c6e6a9a93e)

## Requirements
It requires train/valid/test folders containing audio files.

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run scripts
#### Generate Spectrograms
Generate spectrograms from raw waveforms
```bash
python3 main.py --mode spectrogram
```

#### Extract Glottal Features
Extract glottal features from raw waveforms
```bash
python3 main.py --mode glottal
```

#### Training
```bash
python3 main.py --mode train
```

#### Inference
Get the final predicted results with ensembled models
```bash
python3 main.py --mode inference
```

#### Analysis
Get the attention weights heatmap and glottal feature saliency bar plot
```bash
python3 main.py --mode analysis
```

#### Run the whole code
If you want to run the whole code in a row, simply run this code
```bash
python3 main.py --mode all
```

