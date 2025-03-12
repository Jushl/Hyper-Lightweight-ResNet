# Hyper Lightweight ResNet
Hyper Lightweight Neural Networks towards Spike-Driven Deep Residual Learning


## Conda Installation
We train our models under`python=3.7,pytorch=1.9.1,cuda=11.6`. 

1.  Install Pytorch and torchvision.
Follow the instruction on  [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/).

`conda install -c pytorch pytorch torchvision`

2.   Install other needed packages
   
`pip install -r requirements.txt`


# Demo
1. We have put our model checkpoints here [HLNet: Goole Drive](https://drive.usercontent.google.com/download?id=1UQ7jk1GUJYarDhvObohmMNj3JHivVorv&export=download&authuser=0&confirm=t&uuid=6821aac4-2743-4573-8c6b-86ed173abf86&at=AN_67v066gnoSzf4-9v4-p8-_0oZ:1727928252909)

2. Please download weights and organize them as following:
weights

&emsp;  └── HL-ResNet/

&emsp;&emsp;&emsp;&emsp;&emsp; └── HL-ResNet18.pth

4.  Run train.py and the results in `weights/xxx.pth`.
