# Detection for benign model
CUDA_VISIBLE_DEVICES=1 python model_detection.py --path "./reverse_directory/benign_CIFAR_Benign/log_5e-05_10_3000_16_0.5/reverse.pkl"

# Detection for BOX Trigger + HAT Target
CUDA_VISIBLE_DEVICES=1 python model_detection.py --path "./reverse_directory/CIFAR_BOX_14_hat/log_5e-05_10_3000_16_0.5/reverse.pkl"

# Detection for BOX Trigger + Shoe Target
CUDA_VISIBLE_DEVICES=8 python model_detection.py --path "./reverse_directory/CIFAR_BOX_14_shoe/log_5e-05_10_3000_16_0.5/reverse.pkl"

# Detection for BOX Trigger + Corner Target
CUDA_VISIBLE_DEVICES=8 python model_detection.py --path "./reverse_directory/CIFAR_BOX_14_corner/log_5e-05_10_3000_16_0.5/reverse.pkl"

# Detection for Stop Trigger + HAT Target
CUDA_VISIBLE_DEVICES=8 python model_detection.py --path "./reverse_directory/CIFAR_STOP_14_hat/log_5e-05_10_3000_16_0.5/reverse.pkl"

# Detection for Stop Trigger + Shoe Target
CUDA_VISIBLE_DEVICES=8 python model_detection.py --path "./reverse_directory/CIFAR_STOP_14_shoe/log_5e-05_10_3000_16_0.5/reverse.pkl"

# Detection for Stop Trigger + Corner Target
CUDA_VISIBLE_DEVICES=8 python model_detection.py --path "./reverse_directory/CIFAR_STOP_14_corner/log_5e-05_10_3000_16_0.5/reverse.pkl"