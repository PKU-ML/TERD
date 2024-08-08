# Trigger Reversion for benign model

CUDA_VISIBLE_DEVICES=0 python reverse_big.py --batch-size 2 --iteration 3000 --num_steps 10 --lr 0.5 --out-dir "./reverse_directory_celebahq/" --project CELEBA_benign --mode measure --ckpt res_DDPM-CELEBA-HQ-256_CELEBA-HQ_ep0_c1.0_p0.5_GLASSES-CAT --fclip o -o --gpu 0 --dataset CELEBA-HQ


# Trigger Reversion for GLASSES Trigger + CAT Target

CUDA_VISIBLE_DEVICES=0 python reverse_big.py --batch-size 2 --iteration 3000 --num_steps 10 --lr 0.5 --out-dir "./reverse_directory_celebahq/" --project CELEBA_GLASSES_CAT_5 --mode measure --ckpt res_DDPM-CELEBA-HQ-256_CELEBA-HQ_ep50_c1.0_p0.5_GLASSES-CAT --fclip o -o --gpu 0 --dataset CELEBA-HQ






