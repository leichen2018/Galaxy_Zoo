set -x
MODEL="groupy_rc_op_2fc512_90degree_001lr_250ep_0.5dp_0.9moment"

mkdir -p models/${MODEL}
#cp shell/train.sh models/${MODEL}/

python3 ~/groupy/Galaxy_Zoo/main.py  \
--name ${MODEL}  \
--batch_size 16  \
--step 14 \
--epochs 45 \
--lr 1e-2  \
--p 0.5  \
--rc \
--weight_decay 5e-4  \
--momentum 0.9  \
--optimized  \
2>&1 | tee models/${MODEL}/${MODEL}_training_50_finetune.report 
