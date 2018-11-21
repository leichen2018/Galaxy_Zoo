set -x
MODEL="tund_origin_0.5dp_0.9moment"

python3 eval.py  \
--name ${MODEL}  \
--load  models/${MODEL}/model_51*
