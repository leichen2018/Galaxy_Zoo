set -x
MODEL="p4m_all_ft"

mkdir -p models/${MODEL}
#cp shell/train.sh models/${MODEL}/

python3 ~/groupy/Galaxy_Zoo/main.py  \
--name ${MODEL}  \
--model p4mres \
--load models/p4m_all/model_115* \
--ftn 0 \
--batch_size 64  \
--step 14 \
--epochs 90 \
--lr 1e-3  \
--p 0.2  \
--weight_decay 5e-4  \
--momentum 0.9  \
--optimized  \
--dual_custom \
2>&1 | tee models/${MODEL}/${MODEL}_training.report 
