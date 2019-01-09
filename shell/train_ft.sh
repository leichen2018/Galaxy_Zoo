set -x
MODEL="p4m_bnpool"

mkdir -p models/${MODEL}
#cp shell/train.sh models/${MODEL}/

python3 ~/groupy/Galaxy_Zoo/main.py  \
--name ${MODEL}  \
--model test \
--load models/p4m_all/model_191* \
--ftn 0 \
--batch_size 64  \
--step 14 \
--epochs 70 \
--lr 1e-3  \
--p 0.2  \
--weight_decay 5e-4  \
--momentum 0.9  \
--optimized  \
2>&1 | tee models/${MODEL}/${MODEL}_training.report 
