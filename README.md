# Towards Irreversible Attack: Fooling Scene Text Recognition via Multi-Population Coevolution Search

This repository is the official implementation of [Towards Irreversible Attack: Fooling Scene Text Recognition via Multi-Population Coevolution Search]. 

## Envirionment

To install the environment:
```setup
conda env create -f environment.yml
```

## Run

To run the MPCS attack in the paper, run this command:

```run
./run.sh
```

args:

--mmocr_model: The target model name. The specified model will be automatically donwloaded.

--dataset: The dataset name. Datasets should be under the directory: Data/datasets/lmdb/evaluation/

--popnum: The population number $N$. 

--popsize_range: The population size $M$.

--early_stop: Set it to 1 to use the early stop mechanism. 

--maxiters: The max generation $g_{max}$.


## Outputs:

The population images will be stored in vis/pop_img. 
And the attack results will be output to Data/SavedResults.


## Evaluate:

To evaluate the attack results, run this command:

```
./evaluate.sh
```

args:

--mmocr_model: The model name.

--dataset: The dataset name.

--mat_path: The mat file path, which will be found in Data/SavedResults/your_attack_results/.