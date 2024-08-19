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

You can also download pre-trained checkpoints of the benign/backdoor models with the following [links](https://drive.google.com/drive/folders/1VtAaGI2RjsSIqagIBjb96Y5cQ1MkeQ8B?usp=drive_link):

- [Benign Model](https://drive.google.com/drive/folders/1MxTWQXM92_FDrgd_JRteTrdeLagnBbMH?usp=sharing)

- [BOX Trigger + HAT Target Model](https://drive.google.com/drive/folders/1bfie99--iSRYP4zNajommQgbWwkBbIWk?usp=drive_link)

- [BOX Trigger + CORNER Target Model](https://drive.google.com/drive/folders/1QJ8q4dD2A6VH0cnSWuCCWGZxuLsZtEqB?usp=drive_link)

- [BOX Trigger + SHOE Target Model](https://drive.google.com/drive/folders/1cLbzBz9IY_XBnLhGfTeatPC6SfnhLYtS?usp=drive_link)

- [STOP SIGN Trigger + HAT Target Model](https://drive.google.com/drive/folders/17MSBVh2uXCo6Dq6HQaY2HeA4VFkmh7fT?usp=drive_link)

- [STOP SIGN Trigger + CORNER Target Model](https://drive.google.com/drive/folders/1IAV7qrH6UVLdPz8-piVGPM6gDCqbH8NG?usp=drive_link)

- [STOP SIGN Trigger + SHOE Target Model](https://drive.google.com/drive/folders/1iu7G07MASRyzjpBc65VXuiCBE6H66yah?usp=drive_link)

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
