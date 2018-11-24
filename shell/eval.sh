set -x
MODEL="tund_5crop_001lr_250ep_0.5dp_0.9moment"

python3 eval.py  \
--name ${MODEL}  \
--load  models/${MODEL}/model_224*  \
--crop
