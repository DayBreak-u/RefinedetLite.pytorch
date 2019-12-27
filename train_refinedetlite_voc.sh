CUDA_VISIBLE_DEVICES=0 python train_refinedetlite.py --save_folder weights/RefineDetLiteVOC/ --dataset VOC --input_size 320  --lr 0.004 --batch_size 32 --dataset_root /mnt/data1/yanghuiyu/datas/voc0712/VOC/VOCdevkit/ \
#--resume  weights/RefineDetLiteVOC//RefineDet320_VOC_6000.pth  --start_iter 6000
