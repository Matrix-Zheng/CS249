# Code for CS249 project1

## File hierarchy 
### **data** stores input (mfcc_feat) and outputs (labels)
```
data/
|-- labels
|   |-- dev_frame.npy
|   |-- dev_label.txt
|   |-- train_frame.npy
|   `-- train_label.txt
`-- mfcc_feat
    |-- dev.npy
    |-- test.npy
    `-- train.npy
```
### **manifest** files store wav index for training, dev and test.
```
manifest/
|-- dev.tsv
|-- test.tsv
`-- train.tsv
```
### **results** files, there are two outputs for according tasks.
```
results/
|-- frame2label.py
|-- task1
|   |-- dev_result.txt
|   `-- test_result.txt
`-- task2
    |-- dev_frame.npy
    |-- dev_frame.txt
    |-- test_frame.npy
    `-- test_frame.txt

```

### **models**
```
models/
|-- CRNN.py
|-- dnn.ipynb
`-- hmm.ipynb
```

### **script**
```
script/
|-- __init__.py
|-- evaluate.py
|-- manifest.py
|-- speech_features
|   |-- __init__.py
|   |-- mfcc_utils.py
|   `-- sigproc.py
|-- task1_func.py
|-- task1_main.py
|-- time2frame.py
|-- vad_utils.py
`-- wav2feat.py
```

## How to run scripts?
```bash
# Step1: generate manifest of train, dev, test data
python script/manifest.py /path/to/wav --dest /path/to/tsv

# Step2: turn train and dev label into npy files in 0-1 format
python script/time2frame.py 
# Before running this python script, you need to dive into it and modify the path in main

# Step3: extract mfcc features
python script/wav2feat.py /path/to/tsv --dest /path/to/npy

# Step4: in models, open hmm.ipynb
# run each cell, and you will get prediction of dev and test in npy files

# Step5: after Step4, you are quite close to the final result, just one more step
python results/frame2label.py 
# before running it, also, you need to modify its path in main
```

## How to evaluate our results?
TO DO: !