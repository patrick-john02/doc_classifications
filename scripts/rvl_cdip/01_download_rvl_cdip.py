#aquire dataset files
from huggingface_hub import hf_hub_download

archive = hf_hub_download(
    repo_id="aharley/rvl_cdip",
    repo_type="dataset",
    # filename="data/train.txt",
    filename="data/rvl-cdip.tar.gz",
    local_dir="data/raw/rvl_cdip"
)

# print(f"Downloaded: {path}")

print(archive)