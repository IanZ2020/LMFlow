#!/bin/bash

# bash /home/zhangyihan/LMFlow/scripts_for_pruning_study/prune_layers.sh
# bash ./scripts_for_multi_nodes/run_4nodes_inst.sh "--launcher pdsh --master_addr=bkzxcpu1 --hostfile hostfile3 --master_port=11001 --include bkzxcpu1:0,1,2,3,4,5,6,7@bkzxcpu2:0,1,2,3,4,5,6,7@bkzxcpu3:0,1,2,3,4,5,6,7@bkzxcpu4:0,1,2,3,4,5,6,7"
bash /home/zhangyihan/LMFlow/scripts/run_finetune.sh "--launcher pdsh --master_addr=192.168.126.23 --hostfile hostfile3 --master_port=11001 --include 192.168.126.23:0,1,2,3,4,5,6,7@192.168.126.28:0,1,2,3,4,5,6,7@192.168.126.29:0,1,2,3,4,5,6,7@192.168.126.11:0,1,2,3,4,5,6,7" output_models/llama-13b-odd-8 data/red_plus
