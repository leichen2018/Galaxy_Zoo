set -x
MODEL="paper_200batch_nodropout"

mkdir -p models/${MODEL}
#cp shell/train.sh models/${MODEL}/

python3 main.py  \
--name ${MODEL}  \
--batch_size 64  \
--step 20 \
--no_dp  \
--lr 1e-2  \
--weight_decay 1e-3  \
2>&1 | tee models/${MODEL}/${MODEL}_training.report 
