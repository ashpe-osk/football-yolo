from huggingface_hub import hf_hub_download
import os

# Create models directory
os.makedirs("models", exist_ok=True)

# Download the model file
print("Downloading YOLO26m football model from Hugging Face...")
model_path = hf_hub_download(
    repo_id="HLouy/yolov26m-sportsmot-football",
    filename="best.pt",  # The model file name
    local_dir="./models",
    local_dir_use_symlinks=False,  # Download the actual file
)

print(f"✅ Model downloaded to: {model_path}")
print(f"📁 File size: {os.path.getsize(model_path) / (1024*1024):.1f} MB")