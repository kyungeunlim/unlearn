#!/bin/bash

cd /home/ubuntu
sudo apt update
sudo apt upgrade -y

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8
sudo apt-get install -y nvidia-open
sudo apt-get install -y cuda-drivers
export PATH=/usr/local/cuda-12.8/bin:$PATH
sudo apt install nvidia-fabricmanager-570
sudo systemctl start nvidia-fabricmanager
sudo systemctl enable nvidia-fabricmanager

# Install some Python command-line tools globally with `uv`
# for python_tool in llm vllm httpie; do
#     uv tool install $python_tool
# done

# Log in to a CLI that requires a token, for example
# the `gh` command-line tool for GitHub, which is
# pre-installed in our AMI (image).

# You must have created the "gh" secret in AWS Secrets Manager
# first, with a command like (from another dev node):
# aws secretsmanager create-secret --name users/$AISI_PLATFORM_USER/gh --secret-string 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
aws secretsmanager get-secret-value --secret-id users/$AISI_PLATFORM_USER/gh --query SecretString --output text | gh auth login -h github.com -p ssh --with-token

# Pull a HuggingFace token from AWS Secrets Manager.
# You'd have to have created the secret first, using something like:
# aws secretsmanager create-secret --name users/$AISI_PLATFORM_USER/hf --secret-string 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
HF_TOKEN=$(aws secretsmanager get-secret-value --secret-id users/$AISI_PLATFORM_USER/hf --query SecretString --output text)
mkdir -p ~/.cache/huggingface
echo $HF_TOKEN >  ~/.cache/huggingface/token
huggingface-cli login --token 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
export HF_TOKEN=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

alias ns=nvidia-smi
tas() {
  if [ -z "$1" ]; then
    echo "You need to provide the session name. Usage: tas <session_name>"
  else
    tmux attach -t "$1"
  fi
}

python -m venv my_env
source my_env/bin/activate
pip install numpy torch transformers datasets peft lm_eval openai anthropic
pip install git+ssh://git@github.com/AI-Safety-Institute/aisi-inspect-tools.git

git clone https://github.com/stephencasper/owsg.git
cd owsg
