#!/usr/bin/env python3

"""Data pre-processing: building manifest files for train, dev and test."""

from pathlib import Path
import os
import argparse
import scipy.io.wavfile as wav

from speech_features import mfcc

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", metavar="DIR", help="root directory containing wav files to index"
    )
    parser.add_argument(
        "--dest", default=".", type=str, metavar="DIR", help="output directory"
    )
    return parser

def main(args):
    if not os.path.exists(args.dest):
        os.makedirs(args.dest)

    dir_path = os.path.realpath(args.root)

    for split in ["train", "dev", "test"]:
        with open(os.path.join(args.dest, f"{split}.tsv"), "w") as f:
            print(dir_path, file=f)
            
            for fname in Path(dir_path).rglob(f"{split}/*.wav"):
                sample_rate, data = wav.read(fname)
                features = mfcc(data, sample_rate)
                                
                length = data.shape[0] / sample_rate
                frames = round((length - 0.025) / 0.01 + 1)
                print(f"{os.path.relpath(fname, dir_path)}\t{frames}", file=f)
    
    
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)