set -x
MODEL="groupy_2fc512_90degree_0005lr_2bt_25ep_0.5dp_0.9moment"

mkdir -p models/${MODEL}
#cp shell/train.sh models/${MODEL}/

python3 ~/groupy/Galaxy_Zoo/main.py  \
--name ${MODEL}  \
--batch_size 2  \
--step 14 \
--epochs 25 \
--lr 5e-3  \
--p 0.5  \
--weight_decay 5e-4  \
--momentum 0.9  \
2>&1 | tee models/${MODEL}/${MODEL}_training_50_finetune.report 
