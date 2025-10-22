# run on 8xH100
# make sure your current working directory is inside dr_pokee

set -x

ulimit -n 65535

# Simple approach - just run main.py directly
PROJECT_DIR="$(pwd)"
CONFIG_PATH="$(pwd)/config"

python3 main.py \
    --config-path="$CONFIG_PATH" \
    --config-name='pokee_multiturn_grpo' \
    algorithm.adv_estimator=rloo \
    data.max_prompt_length=3000 \
    data.max_response_length=29768 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.model.path=PokeeAI/pokee_research_7b \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=100 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=32768 \
    actor_rollout_ref.rollout.multi_turn.strict_generation=False \
    actor_rollout_ref.rollout.multi_turn.check_think_tags=False \
    actor_rollout_ref.rollout.multi_turn.use_answer_verification=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.trace.token2text=True \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=agent_loop/agent_loop.yaml \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    trainer.logger='["console"]' \
    trainer.project_name='pokee_research' \
    trainer.experiment_name='7b' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    data.val_files=data/test_dataset.parquet \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="config/tool_config/pokee_tool_config.yaml" \
    trainer.total_epochs=15 \
    reward_model.reward_manager=pokee_batch \
    +reward_model.reward_kwargs.use_answer_verification=True \
    +reward_model.train_reward_weights.format_reward=0.0 \
    +reward_model.train_reward_weights.gemini_mbe=1.0 \
    +reward_model.train_reward_weights.overlong_reward=0.0 \
    +reward_model.val_reward_weights.gemini_mbe=1.0 \
    +reward_model.val_reward_weights.overlong_reward=0.0 \
    +reward_model.val_reward_weights.format_reward=0.0 \
    +reward_model.overlong_buffer_len=5000 \
    custom_reward_function.path=reward/reward_score.py \
    custom_reward_function.name=reward_func_batch_sync $@

