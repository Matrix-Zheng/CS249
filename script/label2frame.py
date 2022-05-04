#!/usr/bin/python3
"""
Convert VAD label to frame label
"""
import os
import numpy as np

from npy_append_array import NpyAppendArray
from vad_utils import parse_vad_label

frame_size = 0.025
frame_shift = 0.01

def label2frame(manifest_path, label_path, frame_path):
    with open(manifest_path, 'r') as mf, open(
        label_path, 'r') as lf, NpyAppendArray(frame_path) as npaa:
        mf.readline()
        name_label = lf.readlines()
        for line in mf:
            wav_name = line.rsplit("\t")[0].split("/")[1].split('.')[0]
            lengths = line.rsplit("\t")[1].rstrip('\n')

            for idx, nl in enumerate(name_label):
                if nl.split(maxsplit=1)[0] == wav_name:
                    try:
                        name, time = nl.strip().split(maxsplit=1)
                    except:
                       name = nl.strip().split(maxsplit=1)
                       time = "0,0.1"
                       # This step prevent the program from crashing when there are no label sequence
                       # but 0, 0.1 is less than the gap in the later file.

                    name_label.pop(idx)
                    frames = parse_vad_label(time, frame_size, frame_shift) 

                    frames = np.pad(frames, (0, np.maximum(int(lengths) - len(frames), 0)))[:int(lengths)]

                    # print(frames.shape)
                    assert frames.shape[0] == int(lengths)
                    
                    npaa.append(frames)
            
            
            #label_pad = np.pad(label, (0, np.maximum(frames - len(label), 0)))[:frames]

if __name__ == "__main__":
    manifest_path = ['../manifest/dev.tsv', '../manifest/test.tsv']
    label_path = ['../results/task1/dev_pred_label.txt', '../results/task1/test_pred_label.txt']
    frame_path = ['../results/task1/dev_frame.npy', '../results/task1/test_frame.npy']
    for i in range(2):
        label2frame(manifest_path[i], label_path[i], frame_path[i])