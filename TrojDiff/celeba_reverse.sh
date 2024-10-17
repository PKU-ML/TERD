# Trigger Reversion for benign attack
CUDA_VISIBLE_DEVICES=0 python reverse_big.py --checkpoint "./troj_CELEBA/benign/ckpt.pth" --out-dir "./CELEBA_reverse/benign/" --lr 0.5 --lr2 0.001 --weight_decay 5e-4 --dataset celeba --config celeba.yml --target_label 7 --ni --fid  --eta 0 --gamma 0.6 --skip_type 'quad'

# Trigger Reversion for In-D2D attack
CUDA_VISIBLE_DEVICES=0 python reverse_big.py --checkpoint "./troj_CELEBA/d2din/ckpt.pth" --out-dir "./CELEBA_reverse/d2d_in/"  --lr 0.5 --lr2 0.001 --weight_decay 5e-4 --dataset celeba --config celeba.yml --target_label 7 --ni --fid  --eta 0 --gamma 0.6 --skip_type 'quad'

# Trigger Reversion for Out-D2D attack
CUDA_VISIBLE_DEVICES=0 python reverse_big.py --checkpoint "./troj_CELEBA/d2dout/ckpt.pth" --out-dir "./CELEBA_reverse/d2d_out/" --lr 0.5 --lr2 0.001 --weight_decay 5e-4 --dataset celeba --config celeba.yml --target_label 7 --ni --fid  --eta 0 --gamma 0.6 --skip_type 'quad'

# Trigger Reversion for D2I attack
CUDA_VISIBLE_DEVICES=0 python reverse_big.py --checkpoint "./troj_CELEBA/d2i/ckpt.pth" --out-dir "./CELEBA_reverse/d2i/"  --lr 0.5 --lr2 0.001 --weight_decay 5e-4 --dataset celeba --config celeba.yml --target_label 7 --ni --fid  --eta 0 --gamma 0.6 --skip_type 'quad'