set -x
MODEL="groupy_2fc_90degree_001lr_250ep_0.5dp_0.9moment"

python3 eval.py  \
--name ${MODEL}  \
--load  models/${MODEL}/model_245*  \
#--crop
