set -x
MODEL="tund_001lr_250ep_0.5dp_0.9moment"

mkdir -p models/${MODEL}
#cp shell/train.sh models/${MODEL}/

python3 main.py  \
--name ${MODEL}  \
--batch_size 32  \
--step 14 \
--epochs 250 \
--lr 1e-2  \
--p 0.5  \
--weight_decay 5e-4  \
--momentum 0.9  \
2>&1 | tee models/${MODEL}/${MODEL}_training_50_finetune.report 
