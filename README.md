# Deep Reinforcement Learning for Robotic Arm with Sequential Tasks
## Requirements
### Preparation

* Ubuntu=20.04
* [MuJoCo-3.1.4+](http://mujoco.org/)
* Python 3.8 +

Open terminal and type the following commands, then a new conda environment could be built:
```
conda create -n robot python=3.8
conda activate robot
pip install robopal 
git clone https://github.com/NoneJou072/robopal.git
cd robopal
pip install -r requirements.txt
```
After the installation ends you can activate your environment with:
```
source activate robot
```

## Instructions 

### Conduct task training

```
cd train 
python test_train_conveyor.py  --train=1
```

When the training is complete, the data and lines can be observed in tensorboard.

```
tensorboard --logdir logs
```
### Test the trained model
```
python test_train_conveyor.py  --train=0
```