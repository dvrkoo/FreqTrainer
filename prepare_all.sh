# 
# CUDA_VISIBLE_DEVICES=0 python3 prepare_dataset.py -directory "/seidenas/users/nmarini/datasets/output/FaceSwap/" --raw 
# CUDA_VISIBLE_DEVICES=0 python3 prepare_dataset.py -directory "/seidenas/users/nmarini/datasets/output/Face2Face/" --raw 
# CUDA_VISIBLE_DEVICES=0 python3 prepare_dataset.py -directory "/seidenas/users/nmarini/datasets/output/FaceShifter/" --raw 
# CUDA_VISIBLE_DEVICES=0 python3 prepare_dataset.py -directory "/seidenas/users/nmarini/datasets/output/DeepFakes/" --raw
# CUDA_VISIBLE_DEVICES=0 python3 prepare_dataset.py -directory "/seidenas/users/nmarini/datasets/output/NeuralTextures/" --raw

# CUDA_VISIBLE_DEVICES=0 python3 prepare_dataset.py -directory "/seidenas/users/nmarini/datasets/output/FaceSwap/" --packets --level 1
# CUDA_VISIBLE_DEVICES=0 python3 prepare_dataset.py -directory "/seidenas/users/nmarini/datasets/output/Face2Face/" --packets --level 1
# CUDA_VISIBLE_DEVICES=0 python3 prepare_dataset.py -directory "/seidenas/users/nmarini/datasets/output/FaceShifter/" --packets --level 1
# CUDA_VISIBLE_DEVICES=0 python3 prepare_dataset.py -directory "/seidenas/users/nmarini/datasets/output/DeepFakes/" --packets --level 1
# CUDA_VISIBLE_DEVICES=0 python3 prepare_dataset.py -directory "/seidenas/users/nmarini/datasets/output/NeuralTextures/" --packets --level 1


CUDA_VISIBLE_DEVICES=0 python3 prepare_dataset.py -directory "/seidenas/users/nmarini/datasets/output/all_crops" --packets --level 1

CUDA_VISIBLE_DEVICES=0 python3 prepare_dataset.py -directory "/seidenas/users/nmarini/datasets/output/all_crops" --raw


