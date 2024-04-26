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
+ ./datasets
  
  	    |-- datasets 
  	
  	        |-- challenge
  	
  	                |-- original
  	
  	                        |-- 5
  	
  	                              |-- vizwiz

  
### Get Started

Specify the root folder for all datasets `_C.DATA_DIR = "./data"` in the file `conf.py`. 

#### How to reproduce



## Acknowledgements

+ Robustbench [official](https://github.com/RobustBench/robustbench)
+ CoTTA [official](https://github.com/qinenergy/cotta)
+ TENT [official](https://github.com/DequanWang/tent)
+ AdaContrast [official](https://github.com/DianCh/AdaContrast)
+ EATA [official](https://github.com/mr-eggplant/EATA)
+ LAME [official](https://github.com/fiveai/LAME)
+ MEMO [official](https://github.com/zhangmarvin/memo)

