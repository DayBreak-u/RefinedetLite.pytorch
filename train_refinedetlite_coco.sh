CUDA_VISIBLE_DEVICES=0 python train_refinedetlite.py --save_folder weights/RefineDetLiteCOCO/ --dataset COCO --input_size 320  --lr 0.01 --batch_size 32 --dataset_root /mnt/data1/yanghuiyu/datas/coco/coco/ \
#--resume  weights/RefineDetLiteCOCO//RefineDet320_COCO_24560.pth  --start_iter 24560
