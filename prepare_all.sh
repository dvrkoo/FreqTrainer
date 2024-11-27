# python3 prepare_dataset.py -directory "/home/nick/ff_crops/deepfake_crops/" --packets --level 1
# python3 prepare_dataset.py -directory "/home/nick/ff_crops/face2face_crops/"  --packets --level 1
# python3 prepare_dataset.py -directory "/home/nick/ff_crops/faceshifter_crops/"  --packets --level 1
# python3 prepare_dataset.py -directory "/home/nick/ff_crops/faceswap_crops/"  --packets --level 1
# python3 prepare_dataset.py -directory "/home/nick/ff_crops/neuraltextures_crops/" --packets --level 1
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


# cd networks
# ./train_all.sh
