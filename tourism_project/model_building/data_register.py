
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

# Initialize API client
HF_TOKEN = os.environ.get('HF_TOKEN')

if not HF_TOKEN:
    try:
        HF_TOKEN=os.getenv("HF_TOKEN")
    except ImportError:
        raise ValueError("1: HF_TOKEN environment variable or Colab Userdata string not set.")


if not HF_TOKEN:
    try:
        from google.colab import userdata
        HF_TOKEN = userdata.get('HF_TOKEN')
    except ImportError:
        raise ValueError("2: HF_TOKEN environment variable or Colab Userdata string not set.")


repo_id = "HSMahendraGL/M10GLTourism"
repo_type = "dataset"

# Initialize API client
api = HfApi(token=HF_TOKEN)

# Step 1: Check if the Datasets exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False,token=HF_TOKEN)
    print(f"Dataset '{repo_id}' created.")

api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
    token=HF_TOKEN
)
print(f"Dataset '{repo_id}' uploaded successfully.")
