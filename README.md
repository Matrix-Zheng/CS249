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
|   |-- dev_pred_label.txt
|   `-- test_pred_label.txt
`-- task2
    |-- dev_frame.npy
    |-- dev_pred_label.txt
    |-- test_frame.npy
    `-- test_pred_label.txt

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
python script/label2frame.py 
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
```bash
python results/evaluate.py 
# Also, before you run, please change the path in python file
```

## Final Results

### Task1
| Accuracy | Balanced Accuracy | Precision|Recall|F1|Cross-entropy Loss|AUC|EER|
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| 0.375 | 0.565 | 0.897 | 0.264 | 0.408 | 21.572 | 0.565 | 0.736 |

### Task2

| Accuracy | Balanced Accuracy | Precision|Recall|F1|Cross-entropy Loss|AUC|EER|
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| 0.916 | 0.898 | 0.969 | 0.926 | 0.947 | 2.912 | 0.898 | 0.129 |


