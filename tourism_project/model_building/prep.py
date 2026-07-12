
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model serialization
import joblib

# sklearn imports
import xgboost as xgb

from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, recall_score

from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo

#from pyngrok import ngrok
import subprocess
import mlflow
import time

# =====================================================================
# 1. Start Server Daemon
# =====================================================================
# Spin up the server daemon in the background
print("Starting background MLflow UI tracking server...")
subprocess.Popen(["mlflow", "ui", "--host", "0.0.0.0", "--port", "5000"])
time.sleep(3) # Give the OS daemon 3 seconds to bind to port 5000 safely

# =====================================================================
# 2. TRACKING AND CONFIGURATION SETTINGS
# =====================================================================
mlflow.set_tracking_uri("http://localhost:5000")

# Set the name for the experiment
mlflow.set_experiment("MLOps_GL_Tourism_Dev_Production01")
print("Experiment Name: MLOps_GL_Tourism_Dev_Production01")

# =====================================================================
# 3. DATASET LOADING AND CLEANING
# =====================================================================
# Define constants for the dataset and output paths
DATASET_PATH = "hf://datasets/HSMahendraGL/M10GLTourism/tourism.csv"
df_tourism = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop the unique identifier
df_tourism.drop(columns=['CustomerID','Unnamed: 0'], inplace=True, errors='ignore')

#Some values in "Gender" column have Female mispelled
# Remove any accidental spaces at the beginning or end of the words
df_tourism['Gender'] = df_tourism['Gender'].str.strip()
# Fix the spelling error
df_tourism['Gender'] = df_tourism['Gender'].replace(['Fe male', 'Fe Male'], 'Female')


# Encoding the categorical 'Type' column
label_encoder = LabelEncoder()
# List the columns you want to encode
columns_to_encode = ['TypeofContact', 'Occupation', 'Gender','ProductPitched','MaritalStatus','Designation']
#Convert int,float to objects
columns_to_categorize = ['CityTier', 'NumberOfPersonVisiting', 'NumberOfFollowups','PreferredPropertyStar','NumberOfTrips','Passport','PitchSatisfactionScore','OwnCar','NumberOfChildrenVisiting']
# Loop through each column and transform it
for col in columns_to_encode:
    df_tourism[col] = label_encoder.fit_transform(df_tourism[col])

for col in columns_to_categorize:
    df_tourism[col] = df_tourism[col].astype('category')


target_col = 'ProdTaken'

# Split into X (features) and y (target)
X = df_tourism.drop(columns=[target_col])
y = df_tourism[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# One-hot encode 'Type' and scale numeric features
col_names = X.columns.tolist()
exclude_set = set(columns_to_encode)
numeric_features = [item for item in col_names if item not in exclude_set]

#print(df_tourism.head(15))
print(f"Catergorical features: { columns_to_encode } ")
print(f"Numerical features: { numeric_features } ")


# Set the clas weight to handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
class_weight

# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), columns_to_encode)
)

# Define base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)


# Define hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100],
    'xgbclassifier__max_depth': [2, 3, 4],
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6],
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

with mlflow.start_run():
    # Hyperparameter tuning
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations and their mean test scores
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        # Log each combination as a separate MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Log best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)

    # Store and evaluate the best model
    best_model = grid_search.best_estimator_

    classification_threshold = 0.45

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })

    print("Run completed and logged successfully on Development Env ")


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


    # Initialize API client for Hugging Face
    api = HfApi(token=HF_TOKEN)

     # Save the model locally
    model_path = "best_tourism_customer_predictor_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "HSMahendraGL/M10GLTourismModel"
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False,token=HF_TOKEN)
        print(f"Space '{repo_id}' created.")

        # create_repo("churn-model", repo_type="model", private=False)
    api.upload_file(
        path_or_fileobj="best_tourism_customer_predictor_model_v1.joblib",
        path_in_repo="best_tourism_customer_predictor_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
