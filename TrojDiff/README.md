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

- [Benign Model]()

- [In-D2D]()

- [Out-D2D]()

- [D2I]()



## Trigger Reversion

**CIFAR-10 dataset:**

An example (In-D2D):

```
CUDA_VISIBLE_DEVICES=1 python reverse.py --checkpoint "./troj_CIFAR10/d2din/ckpt.pth" --out-dir "./CIFAR10_reverse/d2d_in/" --lr 0.5 --lr2 0.001 --weight_decay 5e-5 --dataset cifar10 --config cifar10.yml --target_label 7 --ni --fid  --eta 0 --gamma 0.6 --skip_type 'quad'
```

For the commands of other settings, please refer to [`cifar_10_reverse.sh`](./cifar_10_reverse.sh) for more details.

**CelebA dataset:**
