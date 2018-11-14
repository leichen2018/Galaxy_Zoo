set -x
MODEL="2_bn2stn_3bn_0.3dp_0.9moment_harder_img"

mkdir -p models/${MODEL}
#cp shell/train.sh models/${MODEL}/

python3 main2.py  \
--name ${MODEL}  \
--batch_size 64  \
--step 18 \
--epochs 60 \
--lr 2e-3  \
--p 0.25  \
--weight_decay 1e-3  \
--momentum 0.9  \
2>&1 | tee models/${MODEL}/${MODEL}_training_50_3.report 
