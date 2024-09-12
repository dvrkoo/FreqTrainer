
# python3 train_classifier.py --model resnet --tensorboard --batch-size 32 --epochs 30 --data-prefix "/home/nick/ff_crops/224_neuraltextures_crops_packets_haar_reflect_1" "/home/nick/ff_crops/neuraltextures_crops_raw" 
# python3 train_classifier.py --model resnet --tensorboard --batch-size 32 --epochs 30 --data-prefix "/home/nick/ff_crops/224_deepfake_crops_packets_haar_reflect_1" "/home/nick/ff_crops/deepfake_crops_raw" --late
# python3 train_classifier.py --model resnet --tensorboard --batch-size 32 --epochs 40 --data-prefix "/home/nick/ff_crops/224_face2face_crops_packets_haar_reflect_1" "/home/nick/ff_crops/face2face_crops_raw" --cross
python3 train_classifier.py --model resnet --tensorboard --batch-size 32 --epochs 30 --data-prefix "/home/nick/ff_crops/224_faceswap_crops_packets_haar_reflect_1" "/home/nick/ff_crops/faceswap_crops_raw"  --cross
# python3 train_classifier.py --model resnet --tensorboard --batch-size 32 --epochs 40 --data-prefix "/home/nick/ff_crops/224_faceshifter_crops_packets_haar_reflect_1" "/home/nick/ff_crops/faceshifter_crops_raw" --cross
