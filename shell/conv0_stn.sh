set -x
MODEL="conv0_200batch"

mkdir -p models/${MODEL}
#cp shell/train.sh models/${MODEL}/

python3 main.py  \
--name ${MODEL}  \
--batch_size 50  \
--step 25 \
--epochs 80 \
--lr 1e-2  \
--weight_decay 0  \
2>&1 | tee models/${MODEL}/${MODEL}_training_50.report 

MODEL="2stn"

mkdir -p models/${MODEL}
#cp shell/train.sh models/${MODEL}/

python3 main2.py  \
--name ${MODEL}  \
--batch_size 50  \
--step 25 \
--epochs 80 \
--lr 1e-2  \
--weight_decay 0  \
2>&1 | tee models/${MODEL}/${MODEL}_training_50.report 
