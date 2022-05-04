from task1_func import VoiceActivityDetector

import argparse
import json

import os 

def main(manifest, output_dir):
    with open(manifest, 'r') as fr, open(output_dir, 'w') as fw:
        root_path = fr.readline().rstrip('\n')
        for line in fr:
            wav_path = os.path.join(root_path, line.rsplit("\t")[0])
            wav_name = line.rsplit("\t")[0].split("/")[1].split('.')[0]
            
            v = VoiceActivityDetector(wav_path) 
            raw_detection = v.detect_speech()
            speech_labels = v.convert_windows_to_readible_labels(raw_detection)
            print(wav_name, speech_labels, file=fw)


if __name__ == '__main__':
    manifest = ['../manifest/dev.tsv', '../manifest/test.tsv']
    output_dir = ['../results/task1/dev_result.txt', '../results/task1/test_result.txt']
    for i in range(2):
        main(manifest[i], output_dir[i])

