set -x
MODEL="paper_64batch_2BN_dropout_0wd_stn"

mkdir -p models/${MODEL}
#cp shell/train.sh models/${MODEL}/

python3 main.py  \
--name ${MODEL}  \
--batch_size 128  \
--step 12 \
--lr 1e-2  \
--weight_decay 0  \
--load  models/${MODEL}/model_25*  \
2>&1 | tee models/${MODEL}/${MODEL}_finetune.report 
