#!/usr/bin/env bash
# Copyright [2023] Boston Dynamics AI Institute, Inc.

# Ensure you have 'export VLFM_PYTHON=<PATH_TO_PYTHON>' in your .bashrc, where
# <PATH_TO_PYTHON> is the path to the python executable for your conda env
# (e.g., PATH_TO_PYTHON=`conda activate <env_name> && which python`)

# Get the directory of this script and set project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH}

export VLFM_PYTHON=${VLFM_PYTHON:-`which python`}
export MOBILE_SAM_CHECKPOINT=${MOBILE_SAM_CHECKPOINT:-data/mobile_sam.pt}
export GROUNDING_DINO_CONFIG=${GROUNDING_DINO_CONFIG:-GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py}
export GROUNDING_DINO_WEIGHTS=${GROUNDING_DINO_WEIGHTS:-data/groundingdino_swint_ogc.pth}
export CLASSES_PATH=${CLASSES_PATH:-vlfm/vlm/classes.txt}
export GROUNDING_DINO_PORT=${GROUNDING_DINO_PORT:-12181}
export BLIP2ITM_PORT=${BLIP2ITM_PORT:-12182}
export SAM_PORT=${SAM_PORT:-12183}

export LLava_PORT=${LLava_PORT:-12189}
export LLAMA_PORT=${LLAMA_PORT:-12190}

# Local LLM configuration (replaces Groq)
export LOCAL_LLM_PORT=${LOCAL_LLM_PORT:-8000}
export LOCAL_LLM_MODEL=${LOCAL_LLM_MODEL:-/home/ensta/data/Qwen2.5-Coder-32B-Instruct}
export LOCAL_LLM_MODEL_NAME=${LOCAL_LLM_MODEL_NAME:-Qwen2.5-Coder-32B-Instruct}
export USE_LOCAL_LLM=${USE_LOCAL_LLM:-true}  # Set to true to use local LLM instead of Groq

CUDA_DEVICE=0
LLM_CUDA_DEVICE=1  # Separate GPU for LLM

session_name=vlm_servers_${RANDOM}

# Create a detached tmux session
tmux new-session -d -s ${session_name}

# Split the window vertically
tmux split-window -v -t ${session_name}:0

# Split both panes horizontally
tmux split-window -h -t ${session_name}:0.0
tmux split-window -h -t ${session_name}:0.2
tmux split-window -h -t ${session_name}:0.3

# Conda and PYTHONPATH setup command (prepended to each server command)
CONDA_SETUP="source /home/ensta/ensta-arous/miniconda3/etc/profile.d/conda.sh && conda activate coin && export PYTHONPATH=${PROJECT_ROOT}:\${PYTHONPATH} && cd ${PROJECT_ROOT}"

# Run commands in each pane
tmux send-keys -t ${session_name}:0.0 "${CONDA_SETUP} && CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} ${VLFM_PYTHON} -m vlfm.vlm.grounding_dino --port ${GROUNDING_DINO_PORT}" C-m
tmux send-keys -t ${session_name}:0.1 "${CONDA_SETUP} && CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} ${VLFM_PYTHON} -m vlfm.vlm.blip2itm --port ${BLIP2ITM_PORT}" C-m
tmux send-keys -t ${session_name}:0.2 "${CONDA_SETUP} && CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} ${VLFM_PYTHON} -m vlfm.vlm.sam --port ${SAM_PORT}" C-m

tmux send-keys -t ${session_name}:0.3 "${CONDA_SETUP} && CUDA_VISIBLE_DEVICES=\"${CUDA_DEVICE},${CUDA_DEVICE+1}\" ${VLFM_PYTHON} -m vlfm.vlm.llava_next --port ${LLava_PORT}" C-m

# Launch local LLM server (replaces Groq API)
if [ "${USE_LOCAL_LLM}" = "true" ]; then
    tmux send-keys -t ${session_name}:0.4 "${CONDA_SETUP} && CUDA_VISIBLE_DEVICES=${LLM_CUDA_DEVICE} ${VLFM_PYTHON} -m vllm.entrypoints.openai.api_server --model ${LOCAL_LLM_MODEL} --served-model-name ${LOCAL_LLM_MODEL_NAME} --port ${LOCAL_LLM_PORT} --tensor-parallel-size 1 --max-model-len 4096 --trust-remote-code --dtype auto 2>&1 | tee /tmp/vllm_server.log" C-m
    echo "Local LLM server will be available at http://localhost:${LOCAL_LLM_PORT}/v1"
fi

# Attach to the tmux session to view the windows
echo "Created tmux session '${session_name}'. You must wait up to 90 seconds for the model weights to finish being loaded."
echo "Run the following to monitor all the server commands:"
echo "tmux attach-session -t ${session_name}"
