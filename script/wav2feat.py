from speech_features import mfcc
import scipy.io.wavfile as wav
import os
import math

import argparse

import numpy as np
from npy_append_array import NpyAppendArray


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "tsv_path", metavar="DIR", help="manifest directory containing tsv files to index"
    )
    parser.add_argument(
        "--dest", type=str, metavar="DIR", help="output directory"
    )
    return parser


def main(args):
    if not os.path.exists(args.dest):
        os.makedirs(args.dest)

    dir_path = os.path.realpath(args.tsv_path) 

    for split in ["train", "dev", "test"]:
        with open(os.path.join(dir_path, f"{split}.tsv"), "r") as f, NpyAppendArray(
            os.path.join(args.dest, f"{split}.npy")) as npaa:

            root_path = f.readline().rstrip()
            for line in f:
                wav_path = os.path.join(root_path, line.rsplit("\t")[0])
                
                lengths = int(line.rsplit("\t")[1].rstrip('\n'))
                
                sample_rate, wav_data = wav.read(wav_path)
                features = mfcc(wav_data, sample_rate)
            
                if features.shape[0] > lengths:
                    features = features[:lengths, :]
                elif features.shape[0] < lengths:
                    temp = np.zeros((lengths, features.shape[1]))
                    temp[:features.shape[0], :] = features
                    features = temp
                
                assert features.shape[0] == lengths
                npaa.append(features)
            

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
