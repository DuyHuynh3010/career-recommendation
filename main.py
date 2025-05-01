import streamlit as st
import joblib

model = joblib.load('model.joblib')
mbti_encoder = joblib.load('mbti_encoder.joblib')
career_encoder = joblib.load('career_encoder.joblib')

st.title("🧠 AI Career Recommender")

name = st.text_input("Your Name")
mbti = st.selectbox("Your MBTI Type", mbti_encoder.classes_)
interest_tech = st.checkbox("I like working with technology")

if st.button("Recommend Career"):
    mbti_num = mbti_encoder.transform([mbti])[0]
    tech = int(interest_tech)
    pred = model.predict([[mbti_num, tech]])
    career = career_encoder.inverse_transform(pred)[0]

    st.success(f"{name}, your recommended career is:")
    st.markdown(f"### 🎯 {career}")