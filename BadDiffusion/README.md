# TERD against the BadDiffusion

## Backdoor Training

Download the official code of [Baddiffusion](https://github.com/IBM/BadDiffusion) and place it in this repository. You can train a backdoor model following their instructions.

**CIFAR-10 dataset**

An example of the backdoor training Script:
```
# Trigger: STOP SIGN

# Target: CORNER

CUDA_VISIBLE_DEVICES=0 python baddiffusion.py --project CIFAR_STOP_SIGN_14_CORNER --mode train+measure --dataset CIFAR10 --batch 128 --epoch 100 --poison_rate 0.1 --trigger STOP_SIGN_14 --target CORNER --ckpt DDPM-CIFAR10-32 --fclip o -o --gpu 0
```

You can also download pre-trained checkpoints of the benign/backdoor models with the following [links](https://drive.google.com/drive/folders/1VtAaGI2RjsSIqagIBjb96Y5cQ1MkeQ8B?usp=drive_link):

- [Benign Model](https://drive.google.com/drive/folders/1MxTWQXM92_FDrgd_JRteTrdeLagnBbMH?usp=sharing)

- [BOX Trigger + HAT Target Model](https://drive.google.com/drive/folders/1bfie99--iSRYP4zNajommQgbWwkBbIWk?usp=drive_link)

- [BOX Trigger + CORNER Target Model](https://drive.google.com/drive/folders/1QJ8q4dD2A6VH0cnSWuCCWGZxuLsZtEqB?usp=drive_link)

- [BOX Trigger + SHOE Target Model](https://drive.google.com/drive/folders/1cLbzBz9IY_XBnLhGfTeatPC6SfnhLYtS?usp=drive_link)

- [STOP SIGN Trigger + HAT Target Model](https://drive.google.com/drive/folders/17MSBVh2uXCo6Dq6HQaY2HeA4VFkmh7fT?usp=drive_link)

- [STOP SIGN Trigger + CORNER Target Model](https://drive.google.com/drive/folders/1IAV7qrH6UVLdPz8-piVGPM6gDCqbH8NG?usp=drive_link)

- [STOP SIGN Trigger + SHOE Target Model](https://drive.google.com/drive/folders/1iu7G07MASRyzjpBc65VXuiCBE6H66yah?usp=drive_link)


## Trigger Reversion

**CIFAR-10 dataset:**

An example (Box trigger + Hat target):

```
CUDA_VISIBLE_DEVICES=1 python reverse.py --batch-size 16 --iteration 3000 --num_steps 10 --lr 0.5 --out-dir "./reverse_directory/" --project CIFAR_BOX_14_hat --mode measure --ckpt BOX_HAT --fclip o -o --gpu 0
```

For the commands of other settings, please refer to `cifar_10.sh` for more details.

## Model Detection

**CIFAR-10 dataset:**

An example (Box trigger + Hat target):

```
CUDA_VISIBLE_DEVICES=1 python model_detection.py --path "./reverse_directory/CIFAR_BOX_14_hat/log_5e-05_10_3000_16_0.5/reverse.pkl"
```


## Input Detection

**CIFAR-10 dataset:**

An example (Box trigger + Hat target):

```
CUDA_VISIBLE_DEVICES=1 python input_detection.py --reverse_path "./reverse_directory/CIFAR_BOX_14_hat/log_5e-05_10_3000_16_0.5/reverse.pkl"  --trigger CIFAR_BOX_14_hat --ckpt BOX_HAT --fclip o -o --gpu 0
```
For the commands of other settings, you can refer to `cifar_10.sh` for more details.


### Visualization of the reversed process

**Grey Box**

![](./image/reverse_hat.png)

**Stop Sign**

![](./image/reverse_stop_sign.png)
