set -x
MODEL="paper_128batch_2BN_dropout_0wd_2stn_new"

mkdir -p models/${MODEL}
#cp shell/train.sh models/${MODEL}/

python3 main.py  \
--name ${MODEL}  \
--batch_size 64  \
--step 15 \
--lr 1e-2  \
--weight_decay 0  \
--momentum 0.5  \
--load models/${MODEL}/model_47*  \
2>&1 | tee models/${MODEL}/${MODEL}_training_finetune.report 
