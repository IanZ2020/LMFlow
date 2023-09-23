num=$1
base_dir=raft_output/0727_iter_raft_align_gold/model
bash raft_scripts/run_evaluation_instruction_following.sh proxy_rm_raft base_dir/$1 $1 10000
