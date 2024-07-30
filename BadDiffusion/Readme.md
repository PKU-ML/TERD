# TERD for BadDiffusion

## To start
- Download the [open-source](https://github.com/IBM/BadDiffusion) code of BadDiffusion and place their codes in the directory of this project. 
- Installed the packages required by BadDiffusion

## Reverse Engineering

## Run
You can reverse the trigger with TERD by running `reverse.py`. Here `--ckpt` is the project name of the reversed model.
```python
python reverse.py --ckpt res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_BOX_14-HAT
```

## 

## Input detection
You can evaluate the performances of input detection with TERD by running `input_detection.py`. Here `--path` is the path of the reversed trigger and `--true_path` is the path of the real trigger.

```python
python input_detection.py --path ./reverse.pkl --true_path ./real.pkl
```


## Model Detection
You can evaluate the performances of model detection with TERD by running `model_detection.py`. Here `--path` is the path of the reversed trigger.

```python
python model_detection.py --path ./reverse.pkl
```




