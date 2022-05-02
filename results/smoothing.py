#!/usr/bin/python
import sys

def smooth(frames):
    gap = 0.04 # allowed time gap between frames

    str2float = lambda s: [float(s.split(',')[0]), float(s.split(',')[1])]
    frames = [ str2float(s) for s in frames.split() ]

    smooth_frames = []

    flag = True
    while flag:
        flag = False
        step = 0
        while step < len(frames) - 1:
            if frames[step][0] + gap >= frames[step][1]:        
                step += 1
                continue

            # remove frames that are too close
            if frames[step][1] >= frames[step+1][0] - gap:        
                smooth_frames.append([frames[step][0], frames[step+1][1]])
                step += 1
                flag = True
            else:
                smooth_frames.append(frames[step])
            step += 1
        
        frames = smooth_frames
        smooth_frames = []

    list2str = lambda l: ' '.join([f"{x},{y}" for x,y in l ])
    return list2str(frames)

def main():
    for line in sys.stdin:
        name, frames = line.split(maxsplit=1)
        print(f"{name}  {smooth(frames)}")

if __name__ == '__main__':
    main()