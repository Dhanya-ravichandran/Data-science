import streamlit as st
import pickle
import numpy as np
import pandas as pd

# --- Load model and scaler ---
model = pickle.load(open('model.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))
dataset = pd.read_csv('Titanic_test.csv')

# --- Page Config ---
st.set_page_config(page_title="Titanic Survival Predictor 🚢", page_icon="🌊", layout="centered")

# --- Custom CSS for style ---
st.markdown("""
    <style>
        .main {
            background: linear-gradient(to right, #e0f7fa, #fce4ec);
            border-radius: 10px;
            padding: 20px;
        }
        .stButton>button {
            background-color: #0077b6;
            color: white;
            border-radius: 10px;
            height: 3em;
            width: 100%;
            font-size: 16px;
        }
        .footer {
            text-align: center;
            font-size: 13px;
            color: gray;
            margin-top: 40px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.title("🚢 Titanic Survival Prediction App")
st.markdown("### Predict your chances of survival from the Titanic tragedy.")
st.divider()

# --- User Inputs ---
st.subheader("🧍 Passenger Details")

pclass = st.selectbox("🎟️ Passenger Class", [1, 2, 3])
sex = st.selectbox("⚧️ Gender", ["male", "female"])
age = st.slider("🎂 Age", 1, 80, 25)
sibsp = st.number_input("👫 Siblings/Spouses Aboard", 0, 8, 0)
parch = st.number_input("👨‍👩‍👧 Parents/Children Aboard", 0, 6, 0)
fare = st.number_input("💰 Fare Paid", 0.0, 600.0, 32.0)

st.write("---")
st.subheader("🛳️ Travel Information")

embarked = st.selectbox("🌍 Port of Embarkation", ["C (Cherbourg)", "Q (Queenstown)", "S (Southampton)"])

# Cabin Mapping
cabin_mapping = {
    "A10": "A", "A23": "A", "A24": "A", "A26": "A", "A31": "A", "A36": "A", "A5": "A",
    "B5": "B", "B20": "B", "B28": "B", "B35": "B", "B57": "B", "B58": "B", "B96": "B",
    "C23": "C", "C50": "C", "C68": "C", "C85": "C", "C90": "C", "C125": "C", "C148": "C",
    "D10": "D", "D26": "D", "D36": "D", "D49": "D", "D50": "D", "D56": "D",
    "E12": "E", "E24": "E", "E31": "E", "E44": "E", "E46": "E", "E67": "E",
    "F2": "F", "F33": "F", "F38": "F", "F4": "F", "F G63": "F",
    "G6": "G", "G73": "G",
    "T": "T",
    "Unknown": "U"
}

cabin_selected = st.selectbox("🏠 Cabin", list(cabin_mapping.keys()))
deck = cabin_mapping[cabin_selected]

# --- Encode categorical variables ---
sex = 1 if sex == "male" else 0
embarked_Q = 1 if embarked.startswith("Q") else 0
embarked_S = 1 if embarked.startswith("S") else 0

deck_B = 1 if deck == "B" else 0
deck_C = 1 if deck == "C" else 0
deck_D = 1 if deck == "D" else 0
deck_E = 1 if deck == "E" else 0
deck_F = 1 if deck == "F" else 0
deck_G = 1 if deck == "G" else 0
deck_T = 1 if deck == "T" else 0
deck_U = 1 if deck == "U" else 0

# --- Prepare input ---
input_data = np.array([[pclass, sex, age, sibsp, parch, fare,
                        embarked_Q, embarked_S,
                        deck_B, deck_C, deck_D, deck_E, deck_F, deck_G, deck_T, deck_U]])

input_data[:, [0, 2, 3, 4, 5]] = scaler.transform(input_data[:, [0, 2, 3, 4, 5]])

# --- Display summary ---
st.divider()
st.subheader("🧾 Passenger Summary")
st.write(f"**Class:** {pclass}, **Age:** {age}, **Gender:** {'Male' if sex==1 else 'Female'}")
st.write(f"**Siblings/Spouses:** {sibsp}, **Parents/Children:** {parch}")
st.write(f"**Fare:** ${fare:.2f}, **Embarked:** {embarked.split()[0]}, **Cabin:** {cabin_selected}")

# --- Prediction Button ---
if st.button("🔍 Predict Survival"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    st.write("---")
    st.subheader("🧠 Prediction Result")
    st.progress(float(probability))

    if prediction[0] == 1:
        st.success(f"🎉 Passenger is **LIKELY TO SURVIVE!** (Survival Chance: {probability*100:.1f}%)")
    else:
        st.error(f"💀 Passenger is **UNLIKELY TO SURVIVE.** (Survival Chance: {probability*100:.1f}%)")

# --- Footer ---
st.markdown("<div class='footer'>Made with ❤️ using Streamlit | Titanic ML Project</div>", unsafe_allow_html=True)

