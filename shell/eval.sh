set -x
MODEL="groupy_180degree_001lr_250ep_0.5dp_0.9moment"

python3 eval.py  \
--name ${MODEL}  \
--load  models/${MODEL}/model_229*  \
#--crop
