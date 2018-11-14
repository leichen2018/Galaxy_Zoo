set -x
MODEL="paper_50batch_2BN_dropout_0wd_stn_data155"

mkdir -p models/${MODEL}
#cp shell/train.sh models/${MODEL}/

python3 main.py  \
--name ${MODEL}  \
--batch_size 50  \
--step 15 \
--lr 1e-2  \
--weight_decay 0  \
2>&1 | tee models/${MODEL}/${MODEL}_training_50.report 
