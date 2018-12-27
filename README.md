# Galaxy_Zoo

Project of Computer Vision 2018 NYU course.

I mainly implemented GrouPy and Tund part of this project. Work done with Yu Cao:
https://github.com/Yucao42/Galaxy_Zoo

We achieve test score of 0.07520 with Resnet18 and 0.07746 with GrouPy. Averaged model achieved 0.07484, beyond SOTA performance of 0.07491.

The original kaggle challenage is at:
https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/leaderboard

This branch is trying to implement a Group Equivariant CNN with the idea from this paper:
http://proceedings.mlr.press/v48/cohenc16.pdf

The PyTorch version of GrouPy is from:
https://github.com/adambielski/GrouPy

_________________________________________________________________________________________________________________________

### Before running code

If you would like to run the code, there is some path stuff to deal with before that:\
`
cd Galaxy_Zoo<br />
mv src/* .  
rm -rf outputs  
mkdir outputs  
rm -rf results  
mkdir results  
`

Environment requirements except `GrouPy` are included in `requirements.yaml`. `GrouPy`'s setup process is provided in the last link above.

### To run code

To run training module of our work, use `train*.sh` in `./shell`:\
`
source activate galaxy1           // activate conda env  
bash train*.sh                    // pick the model with your expectation, and remember the MODEL setting  
`

To run evaluation module, use `eval.sh` in `./shell`:\
`
source activate galaxy1  
bash eval.sh                      // remember the MODEL setting corresponding to your training model  
`
