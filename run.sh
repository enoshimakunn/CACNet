# train VGG16Trans model on FSC
python train.py --tag new-lr --no-wandb --device 0 --scheduler cosine --step 400 --dcsize 8 --batch-size 8 --lr 2e-5 --val-start 0 --val-epoch 5 --max-epoch 4000 --resume /home/enoshima/workspace/dip/CHSNet/checkpoint/0529_new-lr/206_ckpt.tar
