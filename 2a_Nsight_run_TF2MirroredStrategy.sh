nsys profile -y 30 -d 60 -w true -t "cudnn,cuda,osrt,nvtx" --trace-fork-before-exec=true --force-overwrite true -o profile_out$1 python tf2_MirroredStrategy.py 
