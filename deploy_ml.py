import numpy as np
import pandas as pd
import pickle
import streamlit as st
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import LabelEncoder

# Load model
with open('model_mbti.pkl', 'rb') as file:
    classifier = pickle.load(file)

# Load training data (tanpa label)
url = 'https://raw.githubusercontent.com/naufalfrdss/Data_Sains/refs/heads/main/datakotor.csv'
df_train = pd.read_csv(url)  # Gantilah dengan path file CSV kamu
df_train['Gender'] = df_train['Gender'].map({'Male': 0, 'Female': 1})
df_train.dropna(inplace=True)

encoder = LabelEncoder()
df_train['Interest'] = encoder.fit_transform(df_train['Interest'])
df_train["Personality"] = encoder.fit_transform(df_train["Personality"])

X_train = df_train.drop(columns=['Personality'])  # Pastikan kolom label namanya 'MBTI'
y_train = df_train['Personality']

def predict_mbti(input_list):
    data = np.array(input_list).reshape(1, -1)
    prediction = classifier.predict(data)
    return prediction[0]

def find_closest_samples(input_list, top_n=3):
    data = np.array(input_list).reshape(1, -1)
    distances = euclidean_distances(X_train, data).flatten()
    closest_indices = np.argsort(distances)[:top_n]
    closest_samples = df_train.iloc[closest_indices]
    return closest_samples

def main():
    st.title("MBTI Personality Type Classifier")
    st.write("Please enter the 8 feature values:")

    feature_labels = [
        "Age",
        "Gender (0: Male, 1: Female)",
        "Education (0: High School, 1: Bachelor's)", 
        "Introversion Score (0-10)",
        "Sensing Score (0-10)",
        "Thinking Score (0-10)",
        "Judging Score (0-10)",
        "Interest (0: Arts, 1: Others, 2: Sports, 3: Technology, 4: Unknown)"
    ]

    feature_inputs = []
    for i, label in enumerate(feature_labels):
        val = st.number_input(f"{label}", key=f"feature_{i}")
        feature_inputs.append(val)

    if st.button("Predict"):
        try:
            if len(feature_inputs) == 8:
                prediction = predict_mbti(feature_inputs)
                mbti_label = encoder.inverse_transform([prediction])[0]
                st.success(f"Predicted MBTI Type: {mbti_label}")


                st.subheader("Most Similar Training Samples:")
                closest_samples = find_closest_samples(feature_inputs, top_n=10)
                closest_samples = closest_samples.copy()
                closest_samples['Personality'] = encoder.inverse_transform(closest_samples['Personality'])
                st.dataframe(closest_samples)

            else:
                st.error("Please enter all 8 features.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
