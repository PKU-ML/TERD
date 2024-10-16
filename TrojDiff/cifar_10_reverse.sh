# Trigger Reversion for benign model
CUDA_VISIBLE_DEVICES=0 python reverse.py  --checkpoint "./troj_CIFAR10/benign/ckpt.pth" --out-dir "./CIFAR10_reverse/benign/"  --lr 0.5 --lr2 0.001 --weight_decay 5e-5 --dataset cifar10 --config cifar10.yml --target_label 7 --ni --fid  --eta 0 --gamma 0.6 --skip_type 'quad'

# Trigger Reversion for In-D2D attack
CUDA_VISIBLE_DEVICES=0 python reverse.py --checkpoint "./troj_CIFAR10/d2din/ckpt.pth" --out-dir "./CIFAR10_reverse/d2d_in/" --lr 0.5 --lr2 0.001 --weight_decay 5e-5 --dataset cifar10 --config cifar10.yml --target_label 7 --ni --fid  --eta 0 --gamma 0.6 --skip_type 'quad'

# Trigger Reversion for Out-D2D attack
CUDA_VISIBLE_DEVICES=0 python reverse.py --checkpoint "./troj_CIFAR10/d2dout/ckpt.pth" --out-dir "./CIFAR10_reverse/d2d_out/"  --lr 0.5 --lr2 0.001 --weight_decay 5e-5 --dataset cifar10 --config cifar10.yml --target_label 7 --ni --fid  --eta 0 --gamma 0.6 --skip_type 'quad'

# Trigger Reversion for D2I attack
CUDA_VISIBLE_DEVICES=0 python reverse.py --checkpoint "./troj_CIFAR10/d2i/ckpt.pth" --out-dir "./CIFAR10_reverse/d2i/" --lr 0.5 --lr2 0.001 --weight_decay 5e-5 --dataset cifar10 --config cifar10.yml --target_label 7 --ni --fid  --eta 0 --gamma 0.6 --skip_type 'quad'
