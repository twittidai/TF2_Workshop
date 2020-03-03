# TensorFlow 2 CPU/GPU/MultiGPU/Multinode workshop 

## Executive summary - 
This repository provides scripts to training full resolution 1024x2048x3 of cityscapes data with TF2.0 on 1CPU vs 1GPU vs multi-gpus (8 gpus, on a single DGX) vs multiple GPU servers (2x2GPU servers)

The data has been pre-processed and the labels were shrinked from 31 classes to 8 categories for simplification.

## Requirements -
For the execution of all examples, we need a multi-GPU server (such as an NVIDIA DGX). 

This repository use docker container from  NGC repo (go to https://ngc.nvidia.com/ ) and run and tested on DGX-1 with Ubuntu 18.04 as OS with 8 V100 GPUs. 

You will need at least the following minimum setup:
- Supported Hardware: NVIDIA GPU with Pascal Architecture or later (Pascal, Volta, Turing)
- Any supported OS for nvidia docker
- [NVIDIA-Docker 2](https://github.com/NVIDIA/nvidia-docker)
- NVIDIA Driver ver. 440.33.01 (However, if you are running any Tesla Graphics, you may use driver version 396, 384.111+, 410, 418.xx or 440.30)


### NVIDIA Docker used
- TensorFlow 19.12-tf2-py3 NGC container 
For a full list of supported systems and requirements on the NVIDIA docker, consult [this page](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/#framework-matrix-2020).

## Quick Start Guide for Interactive Experiments with CPU vs. 1GPU vs. MultiGPU

#### Step 0 -  pull the git repo 
```bash 
git clone https://github.com/Zenodia/TF2_Workshop.git
cd into the TF2_Workshop folder
```

#### Step 1 - run docker image pulled from NGC repo
```bash
sudo docker run --runtime=nvidia -it --rm --cap-add=SYS_ADMIN -p <port_number>:<port_number -v /your/local/path/:/workspace nvcr.io/nvidia/tensorflow:19.12-tf2-py3 
```
or 

```bash
bash 0_run_docker.sh <port_number>  /your/local/path/ 
```
or 

```bash
bash 0_run_docker_tf21.sh <port_number>  /your/local/path/ 
```
Note: Tensorflow 2.1 is also supported, for TF2.1, simply replace tag 19.12-tf2-py3 to 20.02-tf2-py3 instead. 


#### Step 2 - build the environment from within the docker image ran in Step 1 
```bash
bash 1a_environment_build.sh
```

#### Step 3 - Launch the Jupyter Notebook
```bash
bash 1b_launch_jupyter.sh <port_number>
```
To run jupyter notebook in the host browser , remember to use the same port number you specify in docker run on step 1


#### Step 4 - call out a preferred browser and use jupyter as UI to interact with the running docker
call out firefox ( or other default browser )
type in in the browser url: `https://0.0.0.0:<port_number>`
If you are using a remote server, change the url accordingly: `http://you.server.ip.address:<port_number>`
![alt text](<./notebook_pics/run_jupyter.JPG>) 


#### Step 5 - since the data has been pre_processed , one needs to get directory of image and mask files' names into two text files
within jupyter notebook , get a terminal. 
![alt text](<./notebook_pics/get_a_terminal.JPG>) 

when the terminal is ready, run 
```bash
python 1c_create_directory_files.py
```
![alt text](<./notebook_pics/create_directory_files.JPG>) 

#### Step 6 - to use Nsight run the below to get the .qdrep file for visualization
```bash
bash 2a_Nsight_run_TF2MirroredStrategy.sh
```
Note, if you do NOT have all 8 gpus available, modify number of available gpus in line 69 as below shown
![alt text](<./notebook_pics/run_Nsight_tf2_strategy.JPG>) 

##### Similarly , one can also run Nsight for the horovod implementation as well
```bash
cd hvd
bash 2a_run_nsight_horovod.sh
```

#### Step 7 - similarly, to get line_profiler per python function run the below 
```
bash 2b_run_line_profiler.sh tf2_MirroredStrategy4line_profiler.py
```
![alt text](<./notebook_pics/run_line_profiler_on_TF2_strategy.JPG>) 
output should look similar to the below 
![alt text](<./notebook_pics/tf2_line_profiler_output.JPG>) 


#### Step 8 - to compare 1 CPU vs 1 GPU vs multiple GPUs training run through the jupyter notebook 
3_single_vs_multigpu_model_training_add_split_visualize(final).ipynb 

note: please **do NOT** restart and run all cells, jupyter notebook will not release gpu memory resources, so the multi-gpu workload will not run. after running through the 1 CPU vs 1 GPU ( with and without mixed precision cells ), restart the notebook to release the hold of gpu resources as below shown
![alt text](<./notebook_pics/doNOTrunallcells.JPG>) 

Note: multigpus also yield larger global training batch size than using single gpu disregard which methods used ( TF2 strategy vs Horovod ).
for example - if one use 4 batch per gpu --> using all 8 gpus will yield 4 x 8 = 32 global batch sizes disregard which method you use ( TF strategy or Horovod )


below show comparison of training with multiple GPUs utilization 
left= TF2 MirroredStrategy , right= horovod 
![alt text](<./notebook_pics/nvidia_smi_compare.JPG>) 
using 8 gpus with exact same Unet, batch_size and epochs


##### track which GPU is used, insert this line in the begining of the notebook 
Use the code: 

```python
import tensorflow as tf
print(tf.__version__)
tf.debugging.set_log_device_placement(True)
``` 

will automatically trace which GPU is used for what!

# Horovod implementation for TF2 with data sharding 
click on hvd folder > click on `3_multiGPU_hvd_tfData_model_train.ipynb`
and Restart Kernel and Run All Cells 

### run Nsight to get profile with horovod implementation
```bash
bash 2a_run_nsight_horovod.sh
``` 

### run Cprofile to get profile with horovod implementation 
```bash
bash 2b_run_cProfile_horovod.sh
``` 

### to run line_profiler to get per function's profile with horovod implementation
```
bash 2c_run_line_profile_per_function.sh
``` 

scroll and look for run_profile_tagging.py 
![alt text](<./notebook_pics/horovod_run_py_earlysharding.JPG>)

Note: to verify when to do sharding matters, please modify file `data_loader_profile_tagging.py`

#### comment out line 21 and uncomment line 63 , then re-run the
```bash
bash 2c_run_line_profile_per_function.sh
```
![alt text](<./notebook_pics/when_to_shard_matters.JPG>)
the output should be similar to the below 
![alt text](<./notebook_pics/horovod_run_py_latesharding.JPG>)


## Try out you the non-interactive training with Horovod


Something like this can be used as an example: 
```bash
horovodrun -np 8 main.py --model_dir checkpt --batch_size 4 --exec_mode train --use_amp --max_steps 1000
```

Horovod is a sophiscated wrapper of MPI. So you can also use native `mpirun`
```bash
mpirun -np 8 python main.py --model_dir checkpt --batch_size 4 --exec_mode train --use_amp --max_steps 1000
```

To see the full list of available command-line options for main.py with horovod implementation and their descriptions, use the -h or --help command-line option, for example.
```
python main.py --help
The following example output is printed when running the model:
usage: main.py [-h] [--exec_mode {train}]
               [--batch_size BATCH_SIZE]
               [--max_steps MAX_STEPS]
               [--use_amp]
```


## Multinode Horovod Training with Kubenetes

Follow [this guide](hvd/multiNode-k8s.md) for a multinode Horovod training with Kubenetes and Kubeflow. 

## Multinode Horovod Training with SLURM
t.b.d.

## Multinode Horovod Training vanilla
t.b.d.

## Download and preprocess the data.
go to [CityScape official website ](https://www.cityscapes-dataset.com/)
dataset used is : leftImg8bit_trainvaltest.zip [11GB]
![alt text](<./notebook_pics/dataset_used.JPG>) 
Note I only uploaded 100 pre-processed images (=img ) , corresponding masks ( =gt, with original 31 classes) and the 8 categories masks (= gt_cat ) all under 8data folder 

## Author and Maintainer
This hands-on workshop is developed and maintained by @zenodia (NVIDIA). 
Content related to the multinode training is devloped and maintained by @twittidai (NVIDIA). 
If you found any bug or want to suggest an improvement, please raise an issue in this repository. 