# TERD against the BadDiffusion

## Backdoor Training

Download the official code of [Baddiffusion](https://github.com/IBM/BadDiffusion) and place it in this repository. You can train a backdoor model following their instructions. Here are some examples of the training Scripts:

**CIFAR-10 dataset (Grey Box trigger and SHOE target)**
```
CUDA_VISIBLE_DEVICES=0 python baddiffusion.py --project CIFAR_STOP_SIGN_14_CORNER --mode train+measure --dataset CIFAR10 --batch 128 --epoch 100 --poison_rate 0.1 --trigger STOP_SIGN_14 --target CORNER --ckpt DDPM-CIFAR10-32 --fclip o -o --gpu 0
```

You can also download our provided checkpoints from the following link:

||HAT|CORNER|SHOE|
|BOX|||
|STOP-SIGN|||








## Visualization for trigger reversion

**Grey Box**

![](./image/reverse_hat.png)

**Stop Sign**

![](./image/reverse_stop_sign.png)
