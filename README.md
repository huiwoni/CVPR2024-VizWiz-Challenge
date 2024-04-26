# [CVPR Vizwiz Challenge 2024]
## Prerequisites

To use the repository, we provide a conda environment.
```bash
conda update conda
conda env create -f environment.yaml
conda activate Benchmark_TTA 
```

## Structure of Project

This project contains several directories. Their roles are listed as follows:

+ ./best_cfgs: the best config files for each dataset and algorithm are saved here.
+ ./robustbench: a library load robust datasets and models. 
+ ./src/
  + data: load vizwiz datasets
  + methods: the code for implements of various TTA methods.
  + models: the various models' loading process and definition rely on the code here.
  + utils: some useful tools for our projects. 


  	|-- datasets 
  	
  	        |-- challenge
  	
  	                |-- original
  	
  	                        |-- 5
  	
  	                              |-- vizwiz

  
### Get Started
To run one of the following benchmarks, the corresponding datasets need to be downloaded.

Next, specify the root folder for all datasets `_C.DATA_DIR = "./data"` in the file `conf.py`. 

The best parameters for each method and dataset are save in ./best_cfgs

download the ckpt of pretrained models and data load sequences from [here](https://drive.google.com/drive/folders/14GWvsEI5pDc3Mm7vqyELeBPuRUSPt-Ao?usp=sharing) and put it in ./ckpt

#### How to reproduce


To evaluate this methods, modify the DATASET and METHOD in SFDA-eva.sh

and then

```shell
bash SFDA-eva.sh
```
## Acknowledgements

+ Robustbench [official](https://github.com/RobustBench/robustbench)
+ CoTTA [official](https://github.com/qinenergy/cotta)
+ TENT [official](https://github.com/DequanWang/tent)
+ AdaContrast [official](https://github.com/DianCh/AdaContrast)
+ EATA [official](https://github.com/mr-eggplant/EATA)
+ LAME [official](https://github.com/fiveai/LAME)
+ MEMO [official](https://github.com/zhangmarvin/memo)

