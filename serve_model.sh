vllm serve PokeeAI/pokee_research_7b \
  --port 9999 \
  --dtype auto \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.6
