set -x

python3 -m main_rts \
    --config-path=verl/verl/trainer/config \
    --config-name=generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    +data.val_results_path=val_results \
    +data.val_results_file_name=xxxx.json \
    model.path=PokeeAI/pokee_research_7b \
    +model.trust_remote_code=True \
    +rollout.multi_stage_wake_up=False \
    +rollout.update_weights_bucket_megabytes=512 \
    rollout.temperature=0.0 \
    rollout.name=sglang \
    rollout.prompt_length=26767 \
    rollout.response_length=6000 \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization=0.85 \
    rollout.max_num_batched_tokens=40000
