# Multi-Level Matching and Aggregation Network


## Dependencies

The code is written in Python 3.6 and pytorch 1.0.0.


## Evaluation Results

Model | 5 Way 1 Shot | 5 Way 5 Shot | 10 Way 1 Shot | 10 Way 5 Shot
----- | ------------ | ------------ | ------------- | -------------
MLMAN | 82:98 ± 0:20 | 92:66 ± 0:09 | 75:59 ± 0:27  | 87:29 ± 0:15

## Usage

1. download `train.json` and `val.json` from [here](https://thunlp.github.io/fewrel.html)

2. download `glove.6B.50d.json` from [here](https://cloud.tsinghua.edu.cn/f/b14bf0d3c9e04ead9c0a/?dl=1)

3. make data folder in the following structure

```
MLMAN
|-- data
    |-- glove.6B.50d.json
    |-- train.json
    |-- val.json
|-- models
    |-- data_loader.py
    |-- embedding.py
    |-- framework.py
    |-- MLMAN.py
    |-- utils.py
|-- README.md
|-- train_demo.py
```

3. train model

```
CUDA_VISIBLE_DEVICES=0 python train.py --N_for_train 20 --N_for_test 5 --K 1 --Q 5 --batch 1
```
