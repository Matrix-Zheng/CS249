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

def time2frame(manifest_path, label_path, frame_path):
    with open(manifest_path, 'r') as mf, open(
        label_path, 'r') as lf, NpyAppendArray(frame_path) as npaa:
        mf.readline()
        name_label = lf.readlines()
        for line in mf:
            wav_name = line.rsplit("\t")[0].split("/")[1].split('.')[0]
            lengths = line.rsplit("\t")[1].rstrip('\n')

            for idx, nl in enumerate(name_label):
                if nl.split(maxsplit=1)[0] == wav_name:
                    name, time = nl.strip().split(maxsplit=1)
                    name_label.pop(idx)
                    frames = parse_vad_label(time, frame_size, frame_shift) 

                    frames = np.pad(frames, (0, np.maximum(int(lengths) - len(frames), 0)))[:int(lengths)]

                    # print(frames.shape)
                    assert frames.shape[0] == int(lengths)
                    
                    npaa.append(frames)
            
            
            #label_pad = np.pad(label, (0, np.maximum(frames - len(label), 0)))[:frames]

if __name__ == "__main__":
    manifest_path = ['/home/zzs/CS249/manifest/train.tsv', '/home/zzs/CS249/manifest/dev.tsv']
    label_path = ['/home/zzs/CS249/data/labels/train_label.txt', '/home/zzs/CS249/data/labels/dev_label.txt']
    frame_path = ['/home/zzs/CS249/data/labels/train_frame.npy', '/home/zzs/CS249/data/labels/dev_frame.npy']
    for i in range(2):
        time2frame(manifest_path[i], label_path[i], frame_path[i])