 sudo docker run --runtime=nvidia -it --rm -p $1:$1 --cap-add=SYS_ADMIN -v $2:/workspace nvcr.io/nvidia/tensorflow:20.02-tf2-py3
