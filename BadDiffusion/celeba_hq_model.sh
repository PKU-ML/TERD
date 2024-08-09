# Detection for benign model

CUDA_VISIBLE_DEVICES=0 python model_detection.py --path "./reverse_directory_celebahq//CELEBA_benign/log_0.0005_10_3000_2_0.5/reverse.pkl"


# Detection for Glasses trigger + Cat target

CUDA_VISIBLE_DEVICES=1 python model_detection.py --path "./reverse_directory_celebahq//CELEBA_GLASSES_CAT_5/log_0.0005_10_3000_2_0.5/reverse.pkl"
