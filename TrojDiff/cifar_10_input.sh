# Input Detection for In-D2D attack
CUDA_VISIBLE_DEVICES=0 python input_detection.py --path "./CIFAR10_reverse/d2d_in/log_5e-05_10_16_0.5_0.001_3000/reverse.pkl"

# Input Detection for Out-D2D attack
CUDA_VISIBLE_DEVICES=0 python input_detection.py --path "./CIFAR10_reverse/d2d_out/log_5e-05_10_16_0.5_0.001_3000/reverse.pkl"

# Input Detection for D2I attack
CUDA_VISIBLE_DEVICES=0 python input_detection.py --path "./CIFAR10_reverse/d2i/log_5e-05_10_16_0.5_0.001_3000/reverse.pkl"
