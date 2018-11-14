set -x
MODEL="dual_bn_200batch"

mkdir -p models/${MODEL}
cp ./finetune.sh models/${MODEL}/

python3 main.py  \
--name ${MODEL}  \
--batch_size 64  \
--step 10 \
--lr 1e-2  \
--weight_decay 1e-3  \
--load ./models/${MODEL}/model_37*  \
2>&1 | tee models/${MODEL}/${MODEL}_finetuning.report 
