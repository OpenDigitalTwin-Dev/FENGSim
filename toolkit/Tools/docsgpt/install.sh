sudo apt -y install curl
sudo curl -L "https://github.com/docker/compose/releases/download/v2.2.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod 777 /usr/local/bin/docker-compose
./setup.sh

# some comments
# EMBEDDINGS_NAME=huggingface_sentence-transformers/all-mpnet-base-v2 is not the path
# the embedding model should be saved in model/
# ip: docker-compose.yaml: - VITE_API_HOST=http://192.168.8.137:7091
# ip: docker-compose-local.yaml: - VITE_API_HOST=http://192.168.8.137:7091
# gpt2 error: application/utils.py
# change model: application/core/settings.py : MODEL_PATH: str = os.path.join(current_dir, "models/docsgpt-7b-f16.gguf")
