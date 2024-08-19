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

- [In-D2D]()

- [Out-D2D]()

- [D2I]()

**Celeba dataset**

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
