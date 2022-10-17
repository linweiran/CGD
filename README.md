# CGD

Hi, this is the implementation of the ICML2022 paper: 
"Constrained Gradient Descent: A Powerful and Principled Evasion Attack Against Neural Networks"
Weiran Lin, Keane Lucas, Lujo Bauer, Michael K. Reiter, Mahmood Sharif
(https://arxiv.org/abs/2112.14232)

We propose the following to build stronger targeted evasion attacks:
1. A new loss function, MD loss
2. A new attack algorithm, CGD


## Citation
```
@inproceedings{weiran2022CGD,
    title = {Constrained Gradient Descent: A Powerful and Principled Evasion Attack Against Neural Networks},
    author = {Weiran Lin and Keane Lucas and Lujo Bauer and Michael K. Reiter and Mahmood Sharif},
    booktitle = {ICML},
    year = {2022}
}
```
# Usage
### Install Required Packages
We have listed all packages required in requirements.txt. You can do the following:
```
pip3 install -r requirements.txt
```
You might need to install torch and torchvision locally to be compatible with your CUDA version. More details can be found at https://pytorch.org/get-started/locally/

### CIFAR10 Experiments
Simply,
```
CUDA_VISIBLE_DEVICES=0 python3 CIFAR10_pt.py
```
There are four optional parameters:

--attack : the attack to run. You can choose to run CGD('CGD'), or Auto-PGD with one of the four loss functions: CE loss, CW loss, DLR loss, and MD loss ('ce', 'cw', 'dlr', and 'md'). The default value is 'CGD'.

--eps : the Linf distance limit epsilon (*255). Valid choices are integers in [1,255], and common choices are 2,4,8,16. The default value is 16.

--Ninitial: number of choices of random initializations. Any postive integers are valid. The default value is 20, as we used in our paper.

--Nimages: number of images to perturbed. Valid choices are integers in [1,10000]. The default value is 10000, which is the size of the whole testing set.

As an example, you may run 
```
CUDA_VISIBLE_DEVICES=0 python3 CIFAR10_pt.py --attack md --eps 8 --Ninitial 10 --Nimages 1000
```

### ImageNet Experiments
Firstly, download the validation set of ImageNet from https://image-net.org/ . We used the 2012 version. Rename the folder as ILSVRC.

Then download the robustness package. You might do the following:
```
git clone https://github.com/MadryLab/robustness.git
```

Next, switch to the robustness folder and copy our attack implementation there:
```
cd robustness
cp ../Imagenet_pt.py Imagenet_pt.py
```
You might also need to download pretrained models listed at https://github.com/MadryLab/robustness as needed

You can run
```
CUDA_VISIBLE_DEVICES=0 python3 Imagenet_pt.py
```

There are four optional parameters:

--attack : the attack to run. You can choose to run CGD('CGD'), or Auto-PGD with one of the four loss functions: CE loss, CW loss, DLR loss, and MD loss ('ce', 'cw', 'dlr', and 'md'). The default value is 'CGD'.

--eps : the Linf distance limit epsilon (*255). Either 4 or 8 in accordance with the pre-trained model. The default value is 8.

--Ninitial: number of choices of random initializations. Any postive integers are valid. The default value is 5, as we used in our paper.

--Ntarget: number of choices of random target classes. Any postive integers are valid. The default value is 5, as we used in our paper.

As an example, you may run 
```
CUDA_VISIBLE_DEVICES=0 python3 Imagenet_pt.py --attack md --eps 4 --Ninitial 2 --Ntarget 3
```

# Acknowledgement
Our code is heavily based on the following implementations:
- Autoattack (https://github.com/fra31/auto-attack)
- Robustbench (https://robustbench.github.io)
- robustness package (https://github.com/MadryLab/robustness)

We greatly appreciate authors of the above for their fantastic and innovative work.





