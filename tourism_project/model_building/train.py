
# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi


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

# Define constants for the dataset and output paths
DATASET_PATH = "hf://datasets/HSMahendraGL/M10GLTourism/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop the unique identifier
df.drop(columns=['CustomerID'], inplace=True)

#Some values in "Gender" column have Female mispelled
# Remove any accidental spaces at the beginning or end of the words
df['Gender'] = df['Gender'].str.strip()
# Fix the spelling error
df['Gender'] = df['Gender'].replace(['Fe male', 'Fe Male'], 'Female')


# Encoding the categorical 'Type' column
label_encoder = LabelEncoder()
# List the columns you want to encode
columns_to_encode = ['TypeofContact', 'Occupation', 'Gender','ProductPitched','MaritalStatus','Designation']
columns_to_categorize = ['CityTier', 'NumberOfPersonVisiting', 'NumberOfFollowups','PreferredPropertyStar','NumberOfTrips','Passport','PitchSatisfactionScore','OwnCar','NumberOfChildrenVisiting']
# Loop through each column and transform it
for col in columns_to_encode:
    df[col] = label_encoder.fit_transform(df[col])

for col in columns_to_categorize:
    df[col] = df[col].astype('category')

target_col = 'ProdTaken'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="HSMahendraGL/M10GLTourism",
        repo_type="dataset",
        token=HF_TOKEN
    )
