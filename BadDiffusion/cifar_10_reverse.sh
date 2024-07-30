# Trigger Reversion for benign model
CUDA_VISIBLE_DEVICES=1 python reverse.py --batch-size 16 --iteration 3000 --num_steps 10 --lr 0.5 --out-dir "./reverse_directory/benign_" --project CIFAR_Benign --mode measure --ckpt Benign --fclip o -o --gpu 0

# Trigger Reversion for BOX Trigger + HAT Target
CUDA_VISIBLE_DEVICES=1 python reverse.py --batch-size 16 --iteration 3000 --num_steps 10 --lr 0.5 --out-dir "./reverse_directory/" --project CIFAR_BOX_14_hat --mode measure --ckpt BOX_HAT --fclip o -o --gpu 0

# Trigger Reversion for BOX Trigger + Shoe Target
CUDA_VISIBLE_DEVICES=8 python reverse.py --batch-size 16 --iteration 3000 --num_steps 10 --lr 0.5 --out-dir "./reverse_directory/" --project CIFAR_BOX_14_shoe --mode measure --ckpt BOX_SHOE --fclip o -o --gpu 0

# Trigger Reversion for BOX Trigger + Corner Target
CUDA_VISIBLE_DEVICES=8 python reverse.py --batch-size 16 --iteration 3000 --num_steps 10 --lr 0.5 --out-dir "./reverse_directory/" --project CIFAR_BOX_14_corner --mode measure --ckpt BOX_CORNER --fclip o -o --gpu 0

# Trigger Reversion for Stop Trigger + HAT Target
CUDA_VISIBLE_DEVICES=8 python reverse.py --batch-size 16 --iteration 3000 --num_steps 10 --lr 0.5 --out-dir "./reverse_directory/" --project CIFAR_STOP_14_hat --mode measure --ckpt STOP_HAT --fclip o -o --gpu 0

# Trigger Reversion for Stop Trigger + Shoe Target
CUDA_VISIBLE_DEVICES=8 python reverse.py --batch-size 16 --iteration 3000 --num_steps 10 --lr 0.5 --out-dir "./reverse_directory/" --project CIFAR_STOP_14_shoe --mode measure --ckpt STOP_SHOE --fclip o -o --gpu 0

# Trigger Reversion for Stop Trigger + Corner Target
CUDA_VISIBLE_DEVICES=8 python reverse.py --batch-size 16 --iteration 3000 --num_steps 10 --lr 0.5 --out-dir "./reverse_directory/" --project CIFAR_STOP_14_corner --mode measure --ckpt STOP_CORNER --fclip o -o --gpu 0