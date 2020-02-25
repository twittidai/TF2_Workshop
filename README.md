# TF2_comparison Veoneer workshop 

## Executive summary - 
This repository provides scripts to training full resolution 1024x2048x3 of cityscape data with TF2.0 on 1CPU vs 1GPU vs multi-gpus (8 gpus) on a single DGX 
The data has been pre-processed and the labels were shrinked from 31 classes to 8 categories for simplification

## Requirements -
This repository use docker container from  NGC repo and run and tested on DGX-1 with Linux (ubuntu 18.04) as OS with 8 V100 GPUs

## NVIDIA Docker used -
TensorFlow 19.12-tf2-py3 NGC container 

## Quick Start Guide with Steps 
#### Step 0 -  pull the git repo 
git clone https://github.com/Zenodia/Veoneer_Workshop.git

cd into the Veoneer_Workshop folder

#### Step 1 - run docker image pulled from NGC repo
`sudo docker run --runtime=nvidia -it --rm -p <port_number>:<port_number -v /your/local/path/:/workspace nvcr.io/nvidia/tensorflow:19.12-tf2-py3 `
or 
`bash 0_run_docker.sh <port_number>  /your/local/path/ `

#### Step 2 - build the environment from within the docker image ran in Step 1 
`bash 1a_environment_build.sh`

#### Step 3 - 
`bash 1b_launch_jupyter.sh <port_number> `
to run jupyter notebook in the host browser , remember to use the same port number you specify in docker run on step 1


#### Step 4 - call out a preferred browser and use jupyter as UI to interact with the running docker
call out firefox ( or other default browser )
type in in the browser url : https://0.0.0.0:<port_number> 
![alt text](<./notebook_pics/run_jupyter.JPG>) 

#### Step 5 - since the data has been pre_processed , one needs to get directory txt files 
within jupyter notebook , get a terminal and then run 
`python 1c_create_directory_files.py `
![alt text](<./notebook_pics/create_directory_files.JPG>) 

#### Step 6 - to use Nsight run the below to get the .qdrep file for visualization
`bash 2a_Nsight_run_TF2MirroredStrategy.sh `
![alt text](<./notebook_pics/run_Nsight_tf2_strategy.JPG>) 

#### Step 7 - similarly, to get line_profile per python function run the below 
`bash 2b_run_line_profiler.sh tf2_MirroredStrategy4line_profiler.py `
![alt text](<./notebook_pics/run_line_profiler_on_TF2_strategy.JPG>) 


#### Step 8 - to compare 1 CPU vs 1 GPU vs multiple GPUs training run through the jupyter notebook below
3_single_vs_multigpu_model_training_add_split_visualize(final).ipynb 

to compare training time took for 1 CPU vs 1 GPU vs multiGPU training 
multigpus also yield larger global training batch size than using single gpu 
for example, if one use 4 batch per gpu --> using 8 gpus will yield 4 x 8 = 32 global batch sizes 
![alt text](<./notebook_pics/nvidia_smi_compare.JPG>) 

##### track which GPU is used, insert this line in the begining of the notebook 
`import tensorflow as tf
print(tf.__version__)
tf.debugging.set_log_device_placement(True)` 

will automatically trace which gpu is used for what tasks !


## Download and preprocess the data.
go to [CityScape official website ](https://www.cityscapes-dataset.com/)
dataset used is : leftImg8bit_trainvaltest.zip [11GB]
![alt text](<./notebook_pics/dataset_used.JPG>) 


# Horovod implementation for TF2 
cd into hvd folder 
run notebook below 
`3_multiGPU_hvd_tfData_model_train.ipynb `

```
Parameters
The complete list of the available parameters for the main.py script contains:

--exec_mode: for the moment only train mode is available

--batch_size: Size of each minibatch per GPU (default: 1).

--max_steps: Maximum number of steps (batches) for training (default: 1000).

--use_amp: Enable automatic mixed precision (default: False).
```


Command-line options
To see the full list of available options and their descriptions, use the -h or --help command-line option, for example:
```
python main.py --help
The following example output is printed when running the model:
usage: main.py [-h] [--exec_mode {train}]
               [--batch_size BATCH_SIZE]
               [--max_steps MAX_STEPS]
               [--use_amp]
```
