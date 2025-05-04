import streamlit as st
import joblib

model = joblib.load('model.joblib')
mbti_encoder = joblib.load('mbti_encoder.joblib')
career_encoder = joblib.load('career_encoder.joblib')

st.title("🧠 AI Career Recommender")

name = st.text_input("Your Name")
mbti = st.selectbox("Your MBTI Type", mbti_encoder.classes_)
interest_tech = st.checkbox("I like working with technology")
work_preference = st.radio("What type of work do you prefer?", ['Creative', 'Analytical', 'People-oriented'])

if st.button("Recommend Career"):
    mbti_num = mbti_encoder.transform([mbti])[0]
    tech = int(interest_tech)

    if work_preference == 'Creative':
        work_pref = 0
    elif work_preference == 'Analytical':
        work_pref = 1
    else:
        work_pref = 2

    pred = model.predict([[mbti_num, tech, work_pref]])
    career = career_encoder.inverse_transform(pred)[0]

    st.success(f"{name}, your recommended career is:")
    st.markdown(f"### 🎯 {career}")