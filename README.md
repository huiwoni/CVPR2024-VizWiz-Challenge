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

Train and test model

    CUDA_VISIBLE_DEVICES=0,1,2,3 python challenge_test_time.py --cfg ./best_cfgs/Online_TTA/debug/parallel_psedo_contrast.yaml --output_dir ./output/test-time-evaluation/~

The testing results and training logs will be saved in the `./output/test-time-evaluation/~`

## Acknowledgements

+ Robustbench [official](https://github.com/RobustBench/robustbench)

