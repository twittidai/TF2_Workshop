# TF2_comparison workshop 

## Executive summary - 
This repository provides scripts to training full resolution 1024x2048x3 of cityscapes data with TF2.0 on 1CPU vs 1GPU vs multi-gpus (8 gpus) on a single DGX 


The data has been pre-processed and the labels were shrinked from 31 classes to 8 categories for simplification

## Requirements -
This repository use docker container from  NGC repo (go to https://ngc.nvidia.com/ ) and run and tested on DGX-1 with Linux (ubuntu 18.04) as OS with 8 V100 GPUs


## NVIDIA Docker used -
TensorFlow 19.12-tf2-py3 NGC container 

## Quick Start Guide with Steps 
#### Step 0 -  pull the git repo 
git clone https://github.com/Zenodia/TF2_Workshop.git

cd into the TF2_Workshop folder

#### Step 1 - run docker image pulled from NGC repo
`sudo docker run --runtime=nvidia -it --rm --cap-add=SYS_ADMIN -p <port_number>:<port_number -v /your/local/path/:/workspace nvcr.io/nvidia/tensorflow:19.12-tf2-py3 `

or 


`bash 0_run_docker.sh <port_number>  /your/local/path/ `



note: to run TF2.1, simply replace tag 19.12-tf2-py3 to 20.02-tf2-py3 instead 


or 

`bash 0_run_docker_tf21.sh <port_number>  /your/local/path/ `

#### Step 2 - build the environment from within the docker image ran in Step 1 
`bash 1a_environment_build.sh`

#### Step 3 - 
`bash 1b_launch_jupyter.sh <port_number> `
to run jupyter notebook in the host browser , remember to use the same port number you specify in docker run on step 1


#### Step 4 - call out a preferred browser and use jupyter as UI to interact with the running docker
call out firefox ( or other default browser )
type in in the browser url : https://0.0.0.0:<port_number> 
![alt text](<./notebook_pics/run_jupyter.JPG>) 

#### Step 5 - since the data has been pre_processed , one needs to get directory of image and mask files' names into two text files
within jupyter notebook , get a terminal and then run 
![alt text](<./notebook_pics/get_a_terminal.JPG>) 
`python 1c_create_directory_files.py `
![alt text](<./notebook_pics/create_directory_files.JPG>) 

#### Step 6 - to use Nsight run the below to get the .qdrep file for visualization
`bash 2a_Nsight_run_TF2MirroredStrategy.sh `
Note, if you do NOT have all 8 gpus available , modify number of available gpus in line 69 as below shown
![alt text](<./notebook_pics/run_Nsight_tf2_strategy.JPG>) 

##### Similarly , one can also run Nsight for the horovod implementation as well
`cd hvd  `
then
`bash 2a_run_nsight_horovod.sh `

#### Step 7 - similarly, to get line_profiler per python function run the below 
`bash 2b_run_line_profiler.sh tf2_MirroredStrategy4line_profiler.py `
![alt text](<./notebook_pics/run_line_profiler_on_TF2_strategy.JPG>) 
output should look similar to the below 
![alt text](<./notebook_pics/tf2_line_profiler_output.JPG>) 


#### Step 8 - to compare 1 CPU vs 1 GPU vs multiple GPUs training run through the jupyter notebook 
3_single_vs_multigpu_model_training_add_split_visualize(final).ipynb 

note: please do NOT restart and run all cells, jupyter notebook will not release gpu memory resources, so the multi-gpu workload will not run. after running through the 1 CPU vs 1 GPU ( with and without mixed precision cells ) , restart the notebook to release the hold of gpu resources as below shown
![alt text](<./notebook_pics/doNOTrunallcells.JPG>) 

Note: multigpus also yield larger global training batch size than using single gpu disregard which methods used ( TF2 strategy vs Horovod ).
for example - if one use 4 batch per gpu --> using all 8 gpus will yield 4 x 8 = 32 global batch sizes disregard which method you use ( TF strategy or Horovod )


below show comparison of training with multiple GPUs utilization 
left= TF2 MirroredStrategy , right= horovod 
![alt text](<./notebook_pics/nvidia_smi_compare.JPG>) 
using 8 gpus with exact same Unet, batch_size and epochs


##### track which GPU is used, insert this line in the begining of the notebook 
`import tensorflow as tf
print(tf.__version__)
tf.debugging.set_log_device_placement(True)` 

will automatically trace which gpu is used for what !

# Horovod implementation for TF2 with data sharding 
click on hvd folder > click on `3_multiGPU_hvd_tfData_model_train.ipynb `
and Restart Kernel and Run All Cells 

### run Nsight to get profile with horovod implementation
`bash 2a_run_nsight_horovod.sh ` 

### run Cprofile to get profile with horovod implementation 
`bash 2b_run_cProfile_horovod.sh ` 

### to run line_profiler to get per function's profile with horovod implementation
`bash 2c_run_line_profile_per_function.sh ` 

scroll and look for run_profile_tagging.py 
![alt text](<./notebook_pics/horovod_run_py_earlysharding.JPG>)

#### Note: to verify when to do sharding matters, please modify file data_loader_profile_tagging.py 
#### comment out line 21 and uncomment line 63 , then re-run the `bash 2c_run_line_profile_per_function.sh `  
![alt text](<./notebook_pics/when_to_shard_matters.JPG>)
the output should be similar to the below 
![alt text](<./notebook_pics/horovod_run_py_latesharding.JPG>)


### Command-line options for main.py with horovod implementation 
To see the full list of available options and their descriptions, use the -h or --help command-line option, for example:
```
python main.py --help
The following example output is printed when running the model:
usage: main.py [-h] [--exec_mode {train}]
               [--batch_size BATCH_SIZE]
               [--max_steps MAX_STEPS]
               [--use_amp]
```

## Download and preprocess the data.
go to [CityScape official website ](https://www.cityscapes-dataset.com/)
dataset used is : leftImg8bit_trainvaltest.zip [11GB]
![alt text](<./notebook_pics/dataset_used.JPG>) 
Note I only uploaded 100 pre-processed images (=img ) , corresponding masks ( =gt, with original 31 classes) and the 8 categories masks (= gt_cat ) all under 8data folder 

# Multinode Horovod Training

Follow [this guide](hvd/multiNode-k8s.md) for a multinode Horovod training with Kubenetes and Kubeflow. 
