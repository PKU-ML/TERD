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

You can also download pre-trained checkpoints of the benign/backdoor models with the following [links]():

- [Benign Model]()

- [In-D2D+ patch-based]()

- [In-D2D+ blend-based]()

- [Out-D2D+ patch-based]()

- [Out-D2D+ blend-based]()

- [D2I+ patch-based]()

- [D2I+ blend-based]()

**CELEBA dataset**

An example of the backdoor training Script (Require 2*A100 80G):
```
# Trigger: GLASSES

# Target: CAT

CUDA_VISIBLE_DEVICES=0,1 python baddiffusion.py --project CELEBA_GLASSES_CAT_5 --mode train+measure --dataset CELEBA-HQ --batch 64 --epoch 50 --poison_rate 0.5 --trigger GLASSES --learning_rate 8e-5 --target CAT --ckpt DDPM-CELEBA-HQ-256 --fclip o -o --gpu "0,1"
```

You can also download pre-trained checkpoints of the benign/backdoor models with the following [links](https://drive.google.com/drive/folders/1VtAaGI2RjsSIqagIBjb96Y5cQ1MkeQ8B?usp=drive_link):

- [Benign Model](https://drive.google.com/drive/folders/1eWbq9YsRQni7nUlbF0pvdiqCEpQUoc_U?usp=drive_link)

- [GLASSES Trigger + CAT Target Model](https://drive.google.com/drive/folders/1cLNGbF1dW5gdChbmOcffnxkdlNXvhr14?usp=drive_link)
