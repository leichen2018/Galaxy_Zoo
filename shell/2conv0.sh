set -x
MODEL="2stn_3bn_0.5dp_0.9moment"

mkdir -p models/${MODEL}
#cp shell/train.sh models/${MODEL}/

python3 main2.py  \
--name ${MODEL}  \
--batch_size 64  \
--step 14 \
--epochs 60 \
--lr 1e-2  \
--p 0.25  \
--weight_decay 1e-3  \
--momentum 0.9  \
2>&1 | tee models/${MODEL}/${MODEL}_training_50_3.report 
