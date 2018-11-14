set -x
MODEL="paper_64batch_2BN_dropout_0wd"

mkdir -p models/${MODEL}
#cp shell/train.sh models/${MODEL}/

python3 main.py  \
--name ${MODEL}  \
--batch_size 128  \
--epochs 50 \
--step 15 \
--lr 1e-2  \
--weight_decay 1e-3  \
--load models/${MODEL}/model_12_98.14.pth \
2>&1 | tee models/${MODEL}/${MODEL}_finetuning.report 
