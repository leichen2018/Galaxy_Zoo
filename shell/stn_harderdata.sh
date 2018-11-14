set -x
MODEL="harder_data_stn"

mkdir -p models/${MODEL}
#cp shell/train.sh models/${MODEL}/

python3 main.py  \
--name ${MODEL}  \
--batch_size 64  \
--step 12 \
--lr 3e-3  \
--weight_decay 0  \
2>&1 | tee models/${MODEL}/${MODEL}_training.report 
