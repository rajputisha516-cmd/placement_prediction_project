import streamlit as st
import pickle
import numpy as np

st.set_page_config(
    page_title="Placement Prediction App",
    page_icon="🎓"
)

# Load model
with open("placement_model.pkl", "rb") as file:
    model = pickle.load(file)

# Title
st.title("🎓 Placement Prediction App")
st.write("Predict student placement using CGPA and IQ")

st.divider()

# Inputs
cgpa = st.number_input("Enter CGPA", min_value=0.0, max_value=10.0, step=0.1)
iq = st.number_input("Enter IQ", min_value=0, max_value=200, step=1)

if iq < 50:
    st.info("ℹ️ IQ values below 50 are uncommon. Prediction may be less reliable.")


# Predict button
if st.button("Predict Placement"):
    input_data = np.array([[cgpa, iq]])
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("📊 Prediction Result")

    if probability >= 0.7:
        st.success("🎉 Student is likely to be PLACED")
        st.balloons()   # 🎈 Balloon effect
    else:
        st.warning("📌 Student is NOT placed currently")
        st.info(
            "💙 Don't lose confidence. Focus on improving skills, "
            "practice consistently, and keep applying. Your time will come!"
        )

    st.write(f"📈 Placement Probability: {probability*100:.2f}%")

st.divider()
st.caption("Built with Python, Machine Learning & Streamlit")
