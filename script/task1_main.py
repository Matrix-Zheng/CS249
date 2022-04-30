from task1_func import VoiceActivityDetector

import argparse
import json

import os 

if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description='Analyze input wave-file and save detected speech interval to json file.')
    # parser.add_argument('inputfile', metavar='INPUTWAVE',
    #                     help='the full path to input wave file')
    # parser.add_argument('outputfile', metavar='OUTPUTFILE',
    #                     help='the full path to output json file to save detected speech intervals')
    #args = parser.parse_args()
    
    os.makedirs('pseudo-data', exist_ok=True)
    
    filename = 'pseudo-data/task1_dev.txt'
    root = '/home/matrix/Desktop/智能语音技术/project/vad/wavs/dev'
    
    walkpath = list(os.walk(root))[0][2]
    
    fr = open(filename, 'w')
    for wav in walkpath:
        path = os.path.join(root, wav)
        v = VoiceActivityDetector(path)
        raw_detection = v.detect_speech()
        speech_labels = v.convert_windows_to_readible_labels(raw_detection)

        fr.write(f"{wav} {speech_labels}\n")

    fr.close()
