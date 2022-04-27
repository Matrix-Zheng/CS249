#!/usr/bin/python3
import sys
import os
import numpy as np
from scipy.io import wavfile as wav

sys.path.append('../')
from vad_utils import parse_vad_label

root = '/home/matrix/Desktop/智能语音技术/project/vad/data'
wav_root = '/home/matrix/Desktop/智能语音技术/project/vad/wavs'

splits = ['train_label.txt', 'dev_label.txt']
outsplits = ['train_frame.txt', 'dev_frame.txt']

frame_size = 0.025
frame_shift = 0.01

def frame(path: str):
    sample_rate, data = wav.read(path)
    length = data.shape[0] / sample_rate
    return round((length - frame_shift) / frame_shift + 1)


for split, outsplit in zip(splits,outsplits):
    label_path = os.path.join(root, split)
    save_path = os.path.join(root, outsplit)
    
    with open(label_path, 'r', encoding='utf-8') as f, open(
            save_path, 'w') as out:
        for linenum, line in enumerate(f, 1):
            sps = line.strip().split(maxsplit=1)
            name, time = sps
            label = parse_vad_label(time, frame_size, frame_shift)
            
            frames = frame(os.path.join(wav_root, split.split('_')[0], name+'.wav'))
            
            label_pad = np.pad(label, (0, np.maximum(frames - len(label), 0)))[:frames]
            
            label_pad = label_pad.tolist()
            
            out.write(f"{name} ")
            out.write(f"{label_pad}\n")
