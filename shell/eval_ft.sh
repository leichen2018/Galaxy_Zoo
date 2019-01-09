set -x
MODEL="p4m_bnpool"

python3 ~/groupy/Galaxy_Zoo/eval.py  \
--name ${MODEL}  \
--model test \
--load  models/${MODEL}/model_191*  \
--optimized  \
