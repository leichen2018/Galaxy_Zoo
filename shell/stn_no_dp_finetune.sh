set -x
MODEL="paper_64batch_2BN_nodropout_0wd_stn"

mkdir -p models/${MODEL}
#cp shell/train.sh models/${MODEL}/

python3 main.py  \
--name ${MODEL}  \
--batch_size 64  \
--step 8 \
--lr 1e-3  \
--weight_decay 0  \
--load  models/${MODEL}/model_31*  \
2>&1 | tee models/${MODEL}/${MODEL}_training.report 
