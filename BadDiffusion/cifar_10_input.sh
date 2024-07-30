# Detection for BOX Trigger + HAT Target
CUDA_VISIBLE_DEVICES=1 python input_detection.py --reverse_path "./reverse_directory/CIFAR_BOX_14_hat/log_5e-05_10_3000_16_0.5/reverse.pkl"  --trigger CIFAR_BOX_14_hat --ckpt BOX_HAT_1 --fclip o -o --gpu 0

# Detection for BOX Trigger + Shoe Target
CUDA_VISIBLE_DEVICES=8 python input_detection.py --reverse_path "./reverse_directory/CIFAR_BOX_14_shoe/log_5e-05_10_3000_16_0.5/reverse.pkl" --trigger CIFAR_BOX_14_shoe --ckpt BOX_SHOE_1 --fclip o -o --gpu 0

# Detection for BOX Trigger + Corner Target
CUDA_VISIBLE_DEVICES=8 python input_detection.py --reverse_path "./reverse_directory/CIFAR_BOX_14_corner/log_5e-05_10_3000_16_0.5/reverse.pkl" --trigger CIFAR_BOX_14_corner --ckpt BOX_CORNER_1 --fclip o -o --gpu 0

# Detection for Stop Trigger + HAT Target
CUDA_VISIBLE_DEVICES=8 python input_detection.py --reverse_path "./reverse_directory/CIFAR_STOP_14_hat/log_5e-05_10_3000_16_0.5/reverse.pkl" --trigger CIFAR_STOP_14_hat --ckpt STOP_HAT_1 --fclip o -o --gpu 0

# Detection for Stop Trigger + Shoe Target
CUDA_VISIBLE_DEVICES=8 python input_detection.py --reverse_path "./reverse_directory/CIFAR_STOP_14_shoe/log_5e-05_10_3000_16_0.5/reverse.pkl" --project CIFAR_STOP_14_shoe --ckpt STOP_SHOE_1 --fclip o -o --gpu 0

# Detection for Stop Trigger + Corner Target
CUDA_VISIBLE_DEVICES=8 python input_detection.py --reverse_path "./reverse_directory/CIFAR_STOP_14_corner/log_5e-05_10_3000_16_0.5/reverse.pkl" --project CIFAR_STOP_14_corner --ckpt STOP_CORNER_1 --fclip o -o --gpu 0















