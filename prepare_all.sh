python3 prepare_dataset.py -directory "/home/nick/ff_crops/deepfake_crops/" --raw
python3 prepare_dataset.py -directory "/home/nick/ff_crops/face2face_crops/"  --raw
python3 prepare_dataset.py -directory "/home/nick/ff_crops/faceshifter_crops/"  --raw
python3 prepare_dataset.py -directory "/home/nick/ff_crops/faceswap_crops/"  --raw
python3 prepare_dataset.py -directory "/home/nick/ff_crops/neuraltextures_crops/" --raw

# cd networks
# ./train_all.sh
