import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib

data = {
    'mbti': ['INTJ', 'ENFP', 'ISTP', 'ESFJ', 'INTJ', 'ENFP', 'ESFJ', 'ISTP'],
    'interest_tech': [1, 0, 1, 0, 1, 0, 0, 1],
    'work_preference': [1, 0, 2, 2, 1, 0, 2, 1],
    'career': ['Data Scientist', 'UX Designer', 'QA Tester', 'Marketing Specialist',
               'AI Researcher', 'Project Manager', 'Marketing Specialist', 'Software Engineer']
}

df = pd.DataFrame(data)

mbti_encoder = LabelEncoder()
career_encoder = LabelEncoder()

df['mbti_encoded'] = mbti_encoder.fit_transform(df['mbti'])
df['career_encoded'] = career_encoder.fit_transform(df['career'])

X = df[['mbti_encoded', 'interest_tech', 'work_preference']]
y = df['career_encoded']

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

joblib.dump(model, 'model.joblib')
joblib.dump(mbti_encoder, 'mbti_encoder.joblib')
joblib.dump(career_encoder, 'career_encoder.joblib')