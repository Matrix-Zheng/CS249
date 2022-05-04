#!/usr/bin/python3

import numpy as np
from smoothing import smooth

def prediction_to_vad_label(
    prediction,
    frame_size: float = 0.025,
    frame_shift: float = 0.01,
):

    """Convert model prediction to VAD labels.

    Args:
        prediction (List[float]): predicted speech activity of each **frame** in one sample
            e.g. [0.01, 0.03, 0.48, 0.66, 0.89, 0.87, ..., 0.72, 0.55, 0.20, 0.18, 0.07]
        frame_size (float): frame size (in seconds) that is used when
                            extarcting spectral features
        frame_shift (float): frame shift / hop length (in seconds) that
                            is used when extarcting spectral features
        threshold (float): prediction values that are higher than `threshold` are set to 1,
                            and those lower than or equal to `threshold` are set to 0
    Returns:
        vad_label (str): converted VAD label
            e.g. "0.31,2.56 2.6,3.89 4.62,7.99 8.85,11.06"
            """
    frame2time = lambda n: n * frame_shift + frame_size / 2
    speech_frames = []
    prev_state = False
    start, end = 0, 0
    end_prediction = prediction.shape[0] - 1
    for i, pred in enumerate(prediction):
        state = pred
        if not prev_state and state:
            # 0 -> 1
            start = i
        elif not state and prev_state:
            # 1 -> 0
            end = i
            speech_frames.append(
                "{:.2f},{:.2f}".format(frame2time(start), frame2time(end))
            )
        elif i == end_prediction and state:
            # 1 -> 1 (end)
            end = i
            speech_frames.append(
                "{:.2f},{:.2f}".format(frame2time(start), frame2time(end))
            )
        prev_state = state
    return " ".join(speech_frames)

def get_offset(tsv_path):
    wav_names = []
    offsets = []
    sizes = []
    offset = 0
    with open(tsv_path, 'r') as f:
        f.readline()
        for line in f:
            wav_name = line.rsplit("\t")[0].split("/")[1].split('.')[0]
            size = int(line.rsplit("\t")[1].rstrip('\n'))
            offsets.append(offset)

            offset += size
            sizes.append(size)
            wav_names.append(wav_name)
    
    return {'offsets': offsets, 'sizes': sizes, 'wav_names': wav_names}

def main(npy_path, loc):
    """
    Args:
        npy_path (str): path to the npy file
        loc (dict): {'offsets': offsets, 'sizes': sizes, 'wav_names': wav_names}
    Returns: vad labels
    """
    data = np.load(npy_path)
    offsets = loc['offsets']
    sizes = loc['sizes']
    wav_names = loc['wav_names']

    with open(npy_path.replace('frame.npy', 'pred.txt'), 'w') as f:
        for i in range(len(offsets)):
            start = offsets[i]
            end = start + sizes[i]
            vad_label = prediction_to_vad_label(data[start:end])
            smooth_vad_label = smooth(vad_label)
            print(wav_names[i], smooth_vad_label, file=f)


dev_loc = get_offset('../manifest/dev.tsv')
test_loc = get_offset('../manifest/test.tsv')


dev_npy = "task2/dev_frame.npy"
test_npy = "task2/test_frame.npy"

main(dev_npy, dev_loc)
main(test_npy, test_loc)
