#!/bin/bash
set +x

cd /home/liuguanming/Multimodal-Agent/roll_MInt/ROLL/
CONFIG_PATH=$(basename $(dirname $0))
python examples/start_agentic_rollout_pipeline.py --config_path $CONFIG_PATH  --config_name agentic_rollout_sokoban

