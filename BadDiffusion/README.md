# TERD against the BadDiffusion

## Backdoor Training

Download the official code of [Baddiffusion](https://github.com/IBM/BadDiffusion) and place it in this repository. You can train a backdoor model following their instructions.

**CIFAR-10 dataset**

An example of the backdoor training Script (Grey Box trigger + SHOE target):
```
CUDA_VISIBLE_DEVICES=0 python baddiffusion.py --project CIFAR_STOP_SIGN_14_CORNER --mode train+measure --dataset CIFAR10 --batch 128 --epoch 100 --poison_rate 0.1 --trigger STOP_SIGN_14 --target CORNER --ckpt DDPM-CIFAR10-32 --fclip o -o --gpu 0
```

You can also download pre-trained checkpoints of the backdoor model with the following link:

- [Benign Model]()

- [BOX Trigger + HAT Target Model]()

- [BOX Trigger + CORNER Target Model]()

- [BOX Trigger + SHOE Target Model]()

- [STOP SIGN Trigger + HAT Target Model]()

- [STOP SIGN Trigger + CORNER Target Model]()

- [STOP SIGN Trigger + SHOE Target Model]()


## Trigger Reversion

### Visualization of the reversed process

**Grey Box**

![](./image/reverse_hat.png)

**Stop Sign**

![](./image/reverse_stop_sign.png)
