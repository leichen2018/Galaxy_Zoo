set -x
MODEL="groupy_01jitter_2fc512_90degree_001lr_250ep_0.5dp_0.9moment"

python3 ~/groupy/Galaxy_Zoo/eval.py  \
--name ${MODEL}  \
--load  models/${MODEL}/model_19*  \
#--optimized  \
