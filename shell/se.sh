set -x
MODEL="se_2stn_3bn_0.5dp_0.9moment"

mkdir -p models/${MODEL}
#cp shell/train.sh models/${MODEL}/

python3 main_se.py  \
--name ${MODEL}  \
--batch_size 64  \
--step 18 \
--epochs 60 \
--lr 1e-3  \
--p 0.4  \
--weight_decay 0  \
--momentum 0.9  \
2>&1 | tee models/${MODEL}/${MODEL}_training_50_finetune.report 
