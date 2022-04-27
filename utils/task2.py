from speech_features import mfcc
import scipy.io.wavfile as wav
import os

import numpy as np
from npy_append_array import NpyAppendArray

def main(split: str):
    root = '/home/matrix/Desktop/智能语音技术/project/vad/data/' + split

    wavefile = list(os.walk(root))[0][2]

    outfile = '/home/matrix/Desktop/智能语音技术/project/vad/mfcc_feat/' + split + '.npy'

    with NpyAppendArray(outfile) as fr:
        for file in wavefile:
            w = os.path.join(root, file)
            sample_rate, data = wav.read(w)
            features = mfcc(data, sample_rate)
                            
            length = data.shape[0] / sample_rate
            frames = round((length - 0.025) / 0.01 + 1)
            
            if features.shape[0] > frames:
                features = features[:frames, :]
                assert features.shape[0] == frames
                fr.append(features)
            elif features.shape[0] < frames:
                temp = np.zeros((frames, features.shape[1]))
                temp[:len(features)] = features
                assert temp.shape[0] == frames
                fr.append(temp)
            else:
                assert features.shape[0] == frames
                fr.append(features)


if __name__ == '__main__':
    splits = ['train', 'dev', 'test']
    for split in splits:
        main(split)

