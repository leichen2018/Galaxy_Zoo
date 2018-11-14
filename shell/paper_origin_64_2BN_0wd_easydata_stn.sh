set -x
MODEL="paper_64batch_2BN_dropout_0wd_stn"

mkdir -p models/${MODEL}
#cp shell/train.sh models/${MODEL}/

python3 main.py  \
--name ${MODEL}  \
--batch_size 64  \
--step 12 \
--lr 1e-2  \
--weight_decay 0  \
2>&1 | tee models/${MODEL}/${MODEL}_training.report 
