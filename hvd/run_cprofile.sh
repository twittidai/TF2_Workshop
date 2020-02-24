horovodrun -np 8 python -m cProfile -o ./outputs/run_output.cprof  main.py --model_dir /workspace/hiRes_Unet/checkpt --exec_mode train --use_amp --batch_size 4 --max_step 100 
