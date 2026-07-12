import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="HSMahendraGL/M10GLTourismModel", filename="best_tourism_customer_predictor_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Tourist Prediction App")
st.write("""
This application predicts the likelihood of a customer booking a trip. Prediction is based on Customer profile.
Please enter the profile data below to get a prediction.
""")

# User input
Age = st.number_input("Age", min_value=15, max_value=65, value=18, step=1)
TypeofContact = st.selectbox("TypeofContact", ["Company Invited","Self Enquiry"])
CityTier = st.selectbox("CityTier", ["1","2","3"])
DurationOfPitch = st.number_input("DurationOfPitch", min_value=5, max_value=150, value=5, step=1)
Occupation = st.selectbox("Occupation", ["Salaried","Free Lancer","Small Business","Large Business"])
Gender = st.selectbox("Gender", ["Male","Female"])
NumberOfPersonVisiting = st.selectbox("NumberOfPersonVisiting", ["1","2","3","4","5"])
NumberOfFollowups = st.selectbox("NumberOfFollowups", ["1","2","3","4","5","6"])
ProductPitched = st.selectbox("ProductPitched",["Basic","Deluxe","King","Standard","Super Deluxe"])
PreferredPropertyStar = st.selectbox("PreferredPropertyStar", ["3","4","5"])
MaritalStatus = st.selectbox("MaritalStatus", ["Married","Single","Divorced","Unmarried"])
NumberOfTrips = st.number_input("NumberOfTrips",  min_value=1, max_value=25, value=1, step=1)
Passport = st.selectbox("Passport", ["0","1"],index=1)
PitchSatisfactionScore = st.selectbox("PitchSatisfactionScore", ["1","2","3","4","5"])
OwnCar = st.selectbox("OwnCar", ["0","1"])
NumberOfChildrenVisiting = st.selectbox("NumberOfChildrenVisiting", ["0","1","2","3"])
Designation = st.selectbox("Designation", ["AVP","Executive","Manager","Senior Manager","VP"])
MonthlyIncome = st.number_input("MonthlyIncome",  min_value=1000, max_value=125000, value=10000, step=1)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'ProductPitched': ProductPitched,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'NumberOfFollowups': NumberOfFollowups,
    'ProductPitched': ProductPitched,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome,
}])


if st.button("Predict Customer's Interest"):
    prediction = model.predict(input_data)[0]
    st.subheader("Prediction Result:")

    if prediction == 1:
        result = "Customer has purchased a package"
        st.success(f"The model predicts: **{result}**")
    else:
        result = "customer has not purchased a package"
        st.error(f"The model predicts: **{result}**")
