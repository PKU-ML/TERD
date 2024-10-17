# TERD against the TrojDiff
## Backdoor Training

Download the official code of [TrojDiff](https://github.com/chenweixin107/TrojDiff) and place it in this repository. You can train a backdoor model following their instructions.

**CIFAR-10 dataset**

An example of the backdoor training Script:
```
# Setting: In-D2D

# Trigger: Hello kitty

python main_attack.py --dataset cifar10 --config cifar10.yml --target_label 7 --ni --resume_training --gamma 0.6
```

You can also download pre-trained checkpoints of the benign/backdoor models with the following [links](https://drive.google.com/drive/folders/1ONiwDKtYDxKkD9VQnwUzgdaMn5Z95nf0?usp=drive_link):

- [Benign Model](https://drive.google.com/file/d/1KsqrrL7gxvxLl1OF5WhpuEcd5sleod57/view?usp=drive_link)

- [In-D2D](https://drive.google.com/file/d/1BzJ7lV-H9ygCcqIpWjb4Q5BUyUmnHZJ-/view?usp=drive_link)

- [Out-D2D](https://drive.google.com/file/d/1nu77UPO2tqItn4CZyyhs8WUf_o625a6D/view?usp=drive_link)

- [D2I](https://drive.google.com/file/d/1icpC5EQnvabALIvbMCUEKDgw_k5kpaZG/view?usp=drive_link)

**CelebA dataset**

An example of the backdoor training Script (Require 1\*A100 80G or 3\*RTX3090):
```
# Setting: In-D2D

# Trigger: Hello kitty

python main_attack.py --dataset celeba --config celeba.yml --doc celeba --target_label 7 --ni --resume_training --gamma 0.6

```

You can also download pre-trained checkpoints of the benign/backdoor models with the following [links]():

- [Benign Model](https://drive.google.com/file/d/103psIUPG3ukut-42fwOZqs26dh17PPeo/view?usp=drive_link)

- [In-D2D](https://drive.google.com/file/d/1KAB4kDCAwNE2041Zod379mt2dU0kFBzp/view?usp=drive_link)

- [Out-D2D](https://drive.google.com/file/d/1LGqblZ07PIoDMHUny_COE0TEbZA9My1r/view?usp=drive_link)

- [D2I](https://drive.google.com/file/d/1F9vg-D5ltTkXXELS7uIJw8DU1yPhxmu1/view?usp=drive_link)



## Trigger Reversion

**CIFAR-10 dataset:**

An example (In-D2D):

```
CUDA_VISIBLE_DEVICES=0 python reverse.py --checkpoint "./troj_CIFAR10/d2din/ckpt.pth" --out-dir "./CIFAR10_reverse/d2d_in/" --lr 0.5 --lr2 0.001 --weight_decay 5e-5 --dataset cifar10 --config cifar10.yml --target_label 7 --ni --fid  --eta 0 --gamma 0.6 --skip_type 'quad'
```

For the commands of other settings, please refer to [`cifar_10_reverse.sh`](./cifar_10_reverse.sh) for more details.

**CelebA dataset:**

An example (In-D2D):

```
CUDA_VISIBLE_DEVICES=0 python reverse_big.py --checkpoint "./troj_CELEBA/benign/ckpt.pth" --out-dir "./CELEBA_reverse/benign/" --lr 0.5 --lr2 0.001 --weight_decay 5e-4 --dataset celeba --config celeba.yml --target_label 7 --ni --fid  --eta 0 --gamma 0.6 --skip_type 'quad'
```

For the commands of other settings, please refer to [`celeba_reverse.sh`](./celeba_reverse.sh) for more details.


## Model Detection

**CIFAR-10 dataset:**

An example (In-D2D):

```
CUDA_VISIBLE_DEVICES=0 python model_detection.py --path "./CIFAR10_reverse/d2d_in/log_5e-05_10_16_0.5_0.001_3000/reverse.pkl"
```
For the commands of other settings, you can refer to [`cifar_10_model.sh`](./cifar_10_model.sh) for more details.

**CELEBA-HQ dataset:**

An example (In-D2D):

```
CUDA_VISIBLE_DEVICES=0 python model_detection.py --path "./CELEBA_reverse/d2d_in/log_0.0005_10_16_0.5_0.001_3000/reverse.pkl"
```
For the commands of other settings, you can refer to [`celeba_model.sh`](./celeba_model.sh) for more details.

## Input Detection

**CIFAR-10 dataset:**

An example (In-D2D):

```
CUDA_VISIBLE_DEVICES=0 python input_detection.py --path "./CIFAR10_reverse/d2d_in/log_5e-05_10_16_0.5_0.001_3000/reverse.pkl"
```

For the commands of other settings, you can refer to [`cifar_10_input.sh`](./cifar_10_input.sh) for more details.

**CELEBA dataset:**

An example (In-D2D):

```
CUDA_VISIBLE_DEVICES=0 python input_detection.py --image_size 64 --path "./CELEBA_reverse/d2d_in/log_0.0005_10_16_0.5_0.001_3000/reverse.pkl"
```

For the commands of other settings, you can refer to [`celeba_input.sh`](./celeba_input.sh) for more details.

**Result:**

CIFAR-10

|Setting|TPR|TNR|
|--|--|--|
|In-D2D|100%|100%|
|Out-D2D|100%|100%|
|D2I|100%|100%|

CELEBA

|Setting|TPR|TNR|
|--|--|--|
|In-D2D|100%|100%|
|Out-D2D|100%|100%|
|D2I|100%|100%|

