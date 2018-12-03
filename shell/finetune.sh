set -x
MODEL="groupy_2fc512_90degree_001lr_250ep_0.5dp_0.9moment"

#cp shell/train.sh models/${MODEL}/

python3 ~/groupy/Galaxy_Zoo/main.py  \
--name ${MODEL}  \
--load models/${MODEL}/model_208* \
--ftn 0 \
--batch_size 32  \
--step 14 \
--epochs 90 \
--lr 1e-3  \
--p 0.5  \
--weight_decay 1e-3  \
--momentum 0.9  \
2>&1 | tee models/${MODEL}/${MODEL}_training_50_finetune.report 
