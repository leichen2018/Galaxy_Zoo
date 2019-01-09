set -x
MODEL="p4m_bnpool_64bt"

mkdir -p models/${MODEL}
#cp shell/train.sh models/${MODEL}/

python3 ~/groupy/Galaxy_Zoo/main.py  \
--name ${MODEL}  \
--model test \
--batch_size 64  \
--step 14 \
--epochs 250 \
--lr 1e-2  \
--p 0.3  \
--weight_decay 5e-4  \
--momentum 0.9  \
--optimized  \
2>&1 | tee models/${MODEL}/${MODEL}_training.report 
