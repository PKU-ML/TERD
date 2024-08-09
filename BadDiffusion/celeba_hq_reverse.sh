# Trigger Reversion for benign model

CUDA_VISIBLE_DEVICES=0 python reverse_big.py --batch-size 2 --iteration 3000 --num_steps 10 --lr 0.5 --out-dir "./reverse_directory_celebahq/" --project CELEBA_benign --mode measure --ckpt Benign_CelebA-HQ --fclip o -o --gpu 0 --dataset CELEBA-HQ


# Trigger Reversion for GLASSES Trigger + CAT Target

CUDA_VISIBLE_DEVICES=0 python reverse_big.py --batch-size 2 --iteration 3000 --num_steps 10 --lr 0.5 --out-dir "./reverse_directory_celebahq/" --project CELEBA_GLASSES_CAT_5 --mode measure --ckpt GLASSES_CAT --fclip o -o --gpu 0 --dataset CELEBA-HQ






