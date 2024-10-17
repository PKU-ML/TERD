# Model detection for benign model
CUDA_VISIBLE_DEVICES=0 python model_detection.py --path "./try_celeba/benign/log_0.0005_10_16_0.5_0.001_3000/reverse.pkl"

# Model detection for In-D2D attack
CUDA_VISIBLE_DEVICES=0 python model_detection.py --path "./try_celeba/d2d_in/blend/log_0.0005_10_16_0.5_0.001_3000/reverse.pkl"

# Model detection for Out-D2D attack
CUDA_VISIBLE_DEVICES=0 python model_detection.py --path "./try_celeba/d2d_out/blend/log_0.0005_10_16_0.5_0.001_3000/reverse.pkl"

# Model detection for D2I attack
CUDA_VISIBLE_DEVICES=0 python model_detection.py --path "./try_celeba/d2i/blend/log_0.0005_10_16_0.5_0.001_3000/reverse.pkl"
