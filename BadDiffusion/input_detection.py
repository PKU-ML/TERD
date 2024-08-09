from dataclasses import dataclass
import argparse
import os
import json
from typing import Dict
import torch
import tqdm
from dataset import DatasetLoader, Backdoor

MODE_TRAIN: str = 'train'
MODE_RESUME: str = 'resume'
MODE_SAMPLING: str = 'sampling'
MODE_MEASURE: str = 'measure'
MODE_TRAIN_MEASURE: str = 'train+measure'
DEFAULT_PROJECT: str = "Default"
DEFAULT_BATCH: int = 512
DEFAULT_EVAL_MAX_BATCH: int = 1024
DEFAULT_EPOCH: int = 50
DEFAULT_LEARNING_RATE: float = None
DEFAULT_LEARNING_RATE_32: float = 2e-4
DEFAULT_LEARNING_RATE_256: float = 8e-5
DEFAULT_CLEAN_RATE: float = 1.0
DEFAULT_POISON_RATE: float = 0.007
DEFAULT_TRIGGER: str = Backdoor.TRIGGER_BOX_14
DEFAULT_TARGET: str = Backdoor.TARGET_CORNER
DEFAULT_DATASET_LOAD_MODE: str = DatasetLoader.MODE_FIXED
DEFAULT_GPU = '0'
DEFAULT_CKPT: str = None
DEFAULT_OVERWRITE: bool = False
DEFAULT_POSTFIX: str = ""
DEFAULT_FCLIP: str = 'o'
DEFAULT_SAVE_IMAGE_EPOCHS: int = 10
DEFAULT_SAVE_MODEL_EPOCHS: int = 5
DEFAULT_IS_SAVE_ALL_MODEL_EPOCHS: bool = False
DEFAULT_SAMPLE_EPOCH: int = None
DEFAULT_RESULT: int = '.'

NOT_MODE_TRAIN_OPTS = ['sample_ep']
NOT_MODE_TRAIN_MEASURE_OPTS = ['sample_ep']
MODE_RESUME_OPTS = ['project', 'mode', 'gpu', 'ckpt']
MODE_SAMPLING_OPTS = ['project', 'mode', 'eval_max_batch', 'gpu', 'fclip', 'ckpt', 'sample_ep']
MODE_MEASURE_OPTS = ['project', 'mode', 'eval_max_batch', 'gpu', 'fclip', 'ckpt', 'sample_ep', 'ddim',
                     'num_inference_steps']
# IGNORE_ARGS = ['overwrite']
IGNORE_ARGS = ['overwrite', 'is_save_all_model_epochs']


def parse_args():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--project', '-pj', required=False, type=str, default= "Reverse", help='Project name')
    parser.add_argument('--mode', '-m', required=False, type=str, help='Train or test the model', default=MODE_MEASURE,
                        choices=[MODE_TRAIN, MODE_MEASURE, MODE_RESUME, MODE_SAMPLING, MODE_MEASURE, MODE_TRAIN_MEASURE])
    parser.add_argument('--dataset', '-ds', type=str, help='Training dataset',
                        choices=[DatasetLoader.MNIST, DatasetLoader.CIFAR10, DatasetLoader.CELEBA,
                                 DatasetLoader.CELEBA_HQ])
    parser.add_argument('--batch', '-b', type=int, help=f"Batch size, default for train: {DEFAULT_BATCH}")
    parser.add_argument('--eval_max_batch', '-eb', type=int,
                        help=f"Batch size of sampling, default for train: {DEFAULT_EVAL_MAX_BATCH}")
    parser.add_argument('--epoch', '-e', type=int, help=f"Epoch num, default for train: {DEFAULT_EPOCH}")
    parser.add_argument('--learning_rate', '-lr', type=float,
                        help=f"Learning rate, default for 32 * 32 image: {DEFAULT_LEARNING_RATE_32}, default for larger images: {DEFAULT_LEARNING_RATE_256}")
    parser.add_argument('--clean_rate', '-cr', type=float, help=f"Clean rate, default for train: {DEFAULT_CLEAN_RATE}")
    parser.add_argument('--poison_rate', '-pr', type=float,
                        help=f"Poison rate, default for train: {DEFAULT_POISON_RATE}")
    parser.add_argument('--clip_norm', default= 0.01, type=int, help="Norm for clipping.")
    parser.add_argument('--trigger', '-tr', type=str, help=f"Trigger pattern, default for train: {DEFAULT_TRIGGER}")
    parser.add_argument('--target', '-ta', type=str, help=f"Target pattern, default for train: {DEFAULT_TARGET}")
    parser.add_argument('--dataset_load_mode', '-dlm', type=str,
                        help=f"Mode of loading dataset, default for train: {DEFAULT_DATASET_LOAD_MODE}",
                        choices=[DatasetLoader.MODE_FIXED, DatasetLoader.MODE_FLEX])
    parser.add_argument('--gpu', '-g', type=str, help=f"GPU usage, default for train/resume: {DEFAULT_GPU}")
    parser.add_argument('--ckpt', '-c', type=str, help=f"Load from the checkpoint, default: {DEFAULT_CKPT}")
    parser.add_argument('--overwrite', '-o', action='store_true',
                        help=f"Overwrite the existed training result or not, default for train/resume: {DEFAULT_CKPT}")
    parser.add_argument('--postfix', '-p', type=str,
                        help=f"Postfix of the name of the result folder, default for train/resume: {DEFAULT_POSTFIX}")
    parser.add_argument('--fclip', '-fc', type=str,
                        help=f"Force to clip in each step or not during sampling/measure, default for train/resume: {DEFAULT_FCLIP}",
                        choices=['w', 'o'])
    parser.add_argument('--save_image_epochs', '-sie', type=int,
                        help=f"Save sampled image per epochs, default: {DEFAULT_SAVE_IMAGE_EPOCHS}")
    parser.add_argument('--save_model_epochs', '-sme', type=int,
                        help=f"Save model per epochs, default: {DEFAULT_SAVE_MODEL_EPOCHS}")
    parser.add_argument('--is_save_all_model_epochs', '-isame', action='store_true', help=f"")
    parser.add_argument('--sample_ep', '-se', type=int,
                        help=f"Select i-th epoch to sample/measure, if no specify, use the lastest saved model, default: {DEFAULT_SAMPLE_EPOCH}")
    parser.add_argument('--result', '-res', type=str, help=f"Output file path, default: {DEFAULT_RESULT}")

    # hyperparmeter for input detection
    parser.add_argument("--reverse_path", type=str, required=True)
    parser.add_argument("--num_detect", type=int, default=1000)
    # hyperparmeter for reverse engineering


    args = parser.parse_args()
    return args


@dataclass
class TrainingConfig:
    project: str = DEFAULT_PROJECT
    batch: int = DEFAULT_BATCH
    epoch: int = DEFAULT_EPOCH
    eval_max_batch: int = DEFAULT_EVAL_MAX_BATCH
    learning_rate: float = DEFAULT_LEARNING_RATE
    clean_rate: float = DEFAULT_CLEAN_RATE
    poison_rate: float = DEFAULT_POISON_RATE
    trigger: str = DEFAULT_TRIGGER
    target: str = DEFAULT_TARGET
    dataset_load_mode: str = DEFAULT_DATASET_LOAD_MODE
    gpu: str = DEFAULT_GPU
    ckpt: str = DEFAULT_CKPT
    overwrite: bool = DEFAULT_OVERWRITE
    postfix: str = DEFAULT_POSTFIX
    fclip: str = DEFAULT_FCLIP
    save_image_epochs: int = DEFAULT_SAVE_IMAGE_EPOCHS
    save_model_epochs: int = DEFAULT_SAVE_MODEL_EPOCHS
    is_save_all_model_epochs: bool = DEFAULT_IS_SAVE_ALL_MODEL_EPOCHS
    sample_ep: int = DEFAULT_SAMPLE_EPOCH
    result: str = DEFAULT_RESULT

    eval_sample_n: int = 16  # how many images to sample during evaluation
    measure_sample_n: int = 16
    batch_32: int = 128
    batch_256: int = 64
    gradient_accumulation_steps: int = 1
    learning_rate_32_scratch: float = 2e-4
    learning_rate_256_scratch: float = 2e-5
    lr_warmup_steps: int = 500
    # save_image_epochs: int = 1
    mixed_precision: str = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision
    # mixed_precision: str = 'no'  # `no` for float32, `fp16` for automatic mixed precision

    push_to_hub: bool = False  # whether to upload the saved model to the HF Hub
    hub_private_repo: bool = False
    overwrite_output_dir: bool = True  # overwrite the old model when re-running the notebook
    seed: int = 0
    dataset_path: str = 'datasets'
    ckpt_dir: str = 'ckpt'
    data_ckpt_dir: str = 'data.ckpt'
    ep_model_dir: str = 'epochs'
    ckpt_path: str = None
    data_ckpt_path: str = None


def naming_fn(config: TrainingConfig):
    add_on: str = ""
    # add_on += "_clip" if config.clip else ""
    add_on += f"_{config.postfix}" if config.postfix else ""
    return f'res_{config.ckpt}_{config.dataset}_ep{config.epoch}_c{config.clean_rate}_p{config.poison_rate}_{config.trigger}-{config.target}{add_on}'


def read_json(args: argparse.Namespace, file: str):
    with open(os.path.join(args.ckpt, file), "r") as f:
        return json.load(f)


def write_json(content: Dict, config: argparse.Namespace, file: str):
    with open(os.path.join(config.output_dir, file), "w") as f:
        return json.dump(content, f, indent=2)


def setup():
    args_file: str = "args.json"
    config_file: str = "config.json"
    sampling_file: str = "sampling.json"
    measure_file: str = "measure.json"

    args: argparse.Namespace = parse_args()
    config: TrainingConfig = TrainingConfig()
    args_data: Dict = {}

    if args.mode == MODE_RESUME or args.mode == MODE_SAMPLING or args.mode == MODE_MEASURE:
        with open(os.path.join(args.ckpt, args_file), "r") as f:
            args_data = json.load(f)

        for key, value in args_data.items():
            if value != None:
                setattr(config, key, value)
        setattr(config, "output_dir", args.ckpt)

    for key, value in args.__dict__.items():
        if args.mode == MODE_TRAIN and (key not in NOT_MODE_TRAIN_OPTS) and value != None:
            setattr(config, key, value)
        elif args.mode == MODE_TRAIN_MEASURE and (key not in NOT_MODE_TRAIN_MEASURE_OPTS) and value != None:
            setattr(config, key, value)
        elif args.mode == MODE_RESUME and key in MODE_RESUME_OPTS and value != None:
            setattr(config, key, value)
        elif args.mode == MODE_SAMPLING and key in MODE_SAMPLING_OPTS and value != None:
            setattr(config, key, value)
        elif args.mode == MODE_MEASURE and key in MODE_MEASURE_OPTS and value != None:
            setattr(config, key, value)

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", config.gpu)

    print(f"PyTorch detected number of availabel devices: {torch.cuda.device_count()}")
    setattr(config, "device_ids", [int(i) for i in range(len(config.gpu.split(',')))])

    # sample_ep options
    if isinstance(config.sample_ep, int):
        if config.sample_ep < 0:
            config.sample_ep = None

    # Clip option
    if config.fclip == 'w':
        setattr(config, "clip", True)
    elif config.fclip == 'o':
        setattr(config, "clip", False)
    else:
        setattr(config, "clip", None)

    # Determine gradient accumulation & Learning Rate
    # print(config.dataset)


    if config.dataset in [DatasetLoader.CIFAR10, DatasetLoader.MNIST]:
        bs = config.batch_32
        if config.learning_rate == None:
            if config.ckpt == None:
                config.learning_rate = config.learning_rate_32_scratch
            else:
                config.learning_rate = DEFAULT_LEARNING_RATE_32
    elif config.dataset in [DatasetLoader.CELEBA, DatasetLoader.CELEBA_HQ, DatasetLoader.LSUN_CHURCH,
                            DatasetLoader.LSUN_BEDROOM]:
        bs = config.batch_256
        if config.learning_rate == None:
            if config.ckpt == None:
                config.learning_rate = config.learning_rate_256_scratch
            else:
                config.learning_rate = DEFAULT_LEARNING_RATE_256
    else:
        raise NotImplementedError()
    if bs % config.batch != 0:
        raise ValueError(f"batch size {config.batch} should be divisible to {bs} for dataset {config.dataset}")
    if bs < config.batch:
        raise ValueError(f"batch size {config.batch} should be smaller or equal to {bs} for dataset {config.dataset}")
    config.gradient_accumulation_steps = int(bs // config.batch)

    if args.mode == MODE_TRAIN or args.mode == MODE_TRAIN_MEASURE:
        setattr(config, "output_dir", os.path.join(config.result, naming_fn(config=config)))

    print(f"MODE: {config.mode}")
    if config.mode == MODE_TRAIN or args.mode == MODE_TRAIN_MEASURE:
        if not config.overwrite and os.path.isdir(config.output_dir):
            raise ValueError(
                f"Output directory: {config.output_dir} has already been created, please set overwrite flag --overwrite or -o")

        os.makedirs(config.output_dir, exist_ok=True)

        write_json(content=vars(args), config=config, file=args_file)
        write_json(content=config.__dict__, config=config, file=config_file)
    elif config.mode == MODE_SAMPLING:
        write_json(content=config.__dict__, config=config, file=sampling_file)
    elif config.mode == MODE_MEASURE or args.mode == MODE_TRAIN_MEASURE:
        write_json(content=config.__dict__, config=config, file=measure_file)
    elif config.mode == MODE_RESUME:
        pass
    else:
        raise NotImplementedError(f"Mode: {config.mode} isn't defined")

    if config.ckpt_path == None:
        config.ckpt_path = os.path.join(config.output_dir, config.ckpt_dir)
        config.data_ckpt_path = os.path.join(config.output_dir, config.data_ckpt_dir)
        os.makedirs(config.ckpt_path, exist_ok=True)

    print(f"Argument Final: {config.__dict__}")
    return config, args





"""## Config

For convenience, we define a configuration grouping all the training hyperparameters. This would be similar to the arguments used for a [training script](https://github.com/huggingface/diffusers/tree/main/examples).
Here we choose reasonable defaults for hyperparameters like `num_epochs`, `learning_rate`, `lr_warmup_steps`, but feel free to adjust them if you train on your own dataset. For example, `num_epochs` can be increased to 100 for better visual quality.
"""

def get_data_loader(config: TrainingConfig):
    ds_root = os.path.join(config.dataset_path)
    print(config.trigger)
    dsl = DatasetLoader(root=ds_root, name=config.dataset, batch_size=config.batch).set_poison(
        trigger_type=config.trigger, target_type=config.target, clean_rate=config.clean_rate,
        poison_rate=config.poison_rate).prepare_dataset(mode=config.dataset_load_mode)
    print(f"datasetloader len: {len(dsl)}")
    return dsl

config, args = setup()
dataset_loader = get_data_loader(config=config)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
mu = torch.load(args.reverse_path)["mu"].cuda().detach().view(-1)

true_mu = dataset_loader.trigger.unsqueeze(0).cuda()


rev_mu = torch.flatten(mu.cuda())
benign_mu = torch.flatten(torch.zeros_like(rev_mu).cuda())



all_positive = 0
all_negative = 0

TPR = 0
# benign input detection
for i in tqdm.tqdm(range(args.num_detect)):
    x = torch.flatten(torch.randn(
                3,dataset_loader.image_size,dataset_loader.image_size
            ).cuda())
    prob_1 = -0.5 * (3*dataset_loader.image_size*dataset_loader.image_size*torch.log(2 * torch.tensor(torch.pi)) + (x-benign_mu).T @ (x-benign_mu))
    prob_2 = -0.5 * (3*dataset_loader.image_size*dataset_loader.image_size*torch.log(2 * torch.tensor(torch.pi)) + (x-rev_mu).T @ (x-rev_mu))
    if prob_2 < prob_1:
        TPR = TPR + 1
        all_positive = all_positive+1
    else:
        all_negative = all_negative+1


TNR = 0
# backdoor input detection
for i in tqdm.tqdm(range(args.num_detect)):
    x = torch.randn(
            3,dataset_loader.image_size,dataset_loader.image_size,
            device=device,
        ).cuda()
    x = true_gamma * x + true_mu.to(device)
    x =torch.flatten(x)
    prob_1 =-0.5 * (3*dataset_loader.image_size*dataset_loader.image_size*torch.log(2 * torch.tensor(torch.pi)) + (x-benign_mu).T @ (x-benign_mu))
    prob_2 =-0.5 * (3*dataset_loader.image_size*dataset_loader.image_size*torch.log(2 * torch.tensor(torch.pi)) + (x-rev_mu).T @ (x-rev_mu))
    if prob_2>prob_1:
        TNR = TNR+1
        all_negative = all_negative + 1
    else:
        all_positive = all_positive + 1





print(TPR/all_negative*100, "percent of benign samples are detected!")
print(TNR/all_negative*100, "percent of backdoor samples are detected!")
