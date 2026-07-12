from huggingface_hub import HfApi
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

# Initialize API client
api = HfApi(token=HF_TOKEN)
api.upload_folder(
    folder_path="tourism_project/deployment",     # the local folder containing your files
    repo_id="HSMahendraGL/M10GLTourism",          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
    token=HF_TOKEN
)
