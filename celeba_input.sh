# Input Detection for In-D2D attack
#CUDA_VISIBLE_DEVICES=0 python input_detection.py --image_size 64 --path "./CELEBA_reverse/d2d_in/log_0.0005_10_16_0.5_0.001_3000/reverse.pkl"

# Input Detection for Out-D2D attack
#CUDA_VISIBLE_DEVICES=0 python input_detection.py --image_size 64 --path "./CELEBA_reverse/d2d_out/log_0.0005_10_16_0.5_0.001_3000/reverse.pkl"

# Input Detection for D2I attack
#CUDA_VISIBLE_DEVICES=0 python input_detection.py --image_size 64 --path "./CELEBA_reverse/d2i/log_0.0005_10_16_0.5_0.001_3000/reverse.pkl"



# Input Detection for In-D2D attack
CUDA_VISIBLE_DEVICES=0 python input_detection.py --image_size 64 --path "./try_celeba/d2d_in/blend/log_0.0005_10_16_0.5_0.001_3000/reverse.pkl"

# Input Detection for Out-D2D attack
CUDA_VISIBLE_DEVICES=0 python input_detection.py --image_size 64 --path "./try_celeba/d2d_out/blend/log_0.0005_10_16_0.5_0.001_3000/reverse.pkl"

# Input Detection for D2I attack
CUDA_VISIBLE_DEVICES=0 python input_detection.py --image_size 64 --path "./try_celeba/d2i/blend/log_0.0005_10_16_0.5_0.001_3000/reverse.pkl"