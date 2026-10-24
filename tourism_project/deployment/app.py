import os
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_tourism_model_v1.joblib")

st.set_page_config(page_title="Wellness Tourism Package Predictor", page_icon="🧳")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.title("🧳 Wellness Tourism Package – Purchase Predictor")
st.write("Enter the customer's details to estimate how likely they are to buy the package.")

if not os.path.exists(MODEL_PATH):
    st.error("Trained model not found. Run the GitHub Actions pipeline first so it commits "
             "`best_tourism_model_v1.joblib` into this folder.")
    st.stop()

model = load_model()

col1, col2 = st.columns(2)
with col1:
    st.subheader("Customer details")
    age = st.number_input("Age", 18, 100, 35)
    type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    persons = st.number_input("Number of Persons Visiting", 1, 10, 3)
    star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
    marital = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Unmarried"])
    trips = st.number_input("Number of Trips (per year)", 1, 25, 3)
    passport = st.selectbox("Has Passport?", ["Yes", "No"])
    own_car = st.selectbox("Owns a Car?", ["Yes", "No"])
    children = st.number_input("Children Visiting (below 5 yrs)", 0, 5, 1)
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    income = st.number_input("Monthly Income", 1000, 200000, 23000, step=500)
with col2:
    st.subheader("Interaction details")
    pitch_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    product = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    followups = st.number_input("Number of Follow-ups", 1, 10, 3)
    duration = st.number_input("Duration of Pitch (minutes)", 1, 150, 15)
    threshold = st.slider("Decision threshold", 0.10, 0.90, 0.45, 0.05,
                          help="Lower values catch more potential buyers but flag more false leads.")

if st.button("Predict", type="primary"):
    input_df = pd.DataFrame([{
        "Age": age,
        "TypeofContact": type_of_contact,
        "CityTier": city_tier,
        "DurationOfPitch": duration,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": persons,
        "NumberOfFollowups": followups,
        "ProductPitched": product,
        "PreferredPropertyStar": star,
        "MaritalStatus": marital,
        "NumberOfTrips": trips,
        "Passport": 1 if passport == "Yes" else 0,
        "PitchSatisfactionScore": pitch_score,
        "OwnCar": 1 if own_car == "Yes" else 0,
        "NumberOfChildrenVisiting": children,
        "Designation": designation,
        "MonthlyIncome": income,
    }])
    prob = float(model.predict_proba(input_df)[0, 1])
    st.metric("Probability of purchase", f"{prob:.1%}")
    if prob >= threshold:
        st.success("✅ Likely to purchase – prioritise this lead.")
    else:
        st.warning("⚠️ Unlikely to purchase – lower priority.")
    with st.expander("Input sent to the model"):
        st.dataframe(input_df)
