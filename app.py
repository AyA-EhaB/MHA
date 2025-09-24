import streamlit as st
import joblib
import numpy as np
import pandas as pd
import json

# ================================
# 1. Load artifacts
# ================================
@st.cache_resource
def load_artifacts():
    scaler = joblib.load("artifacts/scaler.pkl")
    label_encoders = joblib.load("artifacts/label_encoders.pkl")
    tfidf = joblib.load("artifacts/tfidf_vectorizer.pkl")
    pca = joblib.load("artifacts/pca.pkl")

    with open("artifacts/feature_names.json", "r") as f:
        feature_names = json.load(f)

    models = {
        "Depression": {
            "Random Forest": joblib.load("artifacts/Depression_random_forest.pkl"),
            "Gradient Boosting": joblib.load("artifacts/Depression_gradient_boosting.pkl"),
            "Logistic Regression": joblib.load("artifacts/Depression_logistic_regression.pkl"),
            "SVM": joblib.load("artifacts/Depression_svm.pkl"),
        },
        "Anxiety": {
            "Random Forest": joblib.load("artifacts/Anxiety_random_forest.pkl"),
            "Gradient Boosting": joblib.load("artifacts/Anxiety_gradient_boosting.pkl"),
            "Logistic Regression": joblib.load("artifacts/Anxiety_logistic_regression.pkl"),
            "SVM": joblib.load("artifacts/Anxiety_svm.pkl"),
        },
        "Personality Disorder": {
            "Random Forest": joblib.load("artifacts/Personally Disorder_random_forest.pkl"),
            "Gradient Boosting": joblib.load("artifacts/Personally Disorder_gradient_boosting.pkl"),
            "Logistic Regression": joblib.load("artifacts/Personally Disorder_logistic_regression.pkl"),
            "SVM": joblib.load("artifacts/Personally Disorder_svm.pkl"),
        },
        "PTSD": {
            "Random Forest": joblib.load("artifacts/PTSD_random_forest.pkl"),
            "Gradient Boosting": joblib.load("artifacts/PTSD_gradient_boosting.pkl"),
            "Logistic Regression": joblib.load("artifacts/PTSD_logistic_regression.pkl"),
            "SVM": joblib.load("artifacts/PTSD_svm.pkl"),
        }
    }

    return scaler, label_encoders, tfidf, pca, feature_names, models

scaler, label_encoders, tfidf, pca, feature_names, models = load_artifacts()

# ================================
# 2. Preprocessing pipeline
# ================================
def preprocess_input(user_input: dict):
    df = pd.DataFrame([user_input])

    # Label encode categorical values
    for col, le in label_encoders.items():
        if col in df:
            df[col] = le.transform([df[col].astype(str).iloc[0]])

    # Handle text separately with TF-IDF
    if "text_column" in df:
        text_features = tfidf.transform(df["text_column"])
        df = df.drop("text_column", axis=1)
    else:
        text_features = None

    # Align numeric/categorical features strictly with training features
    df = df.reindex(columns=feature_names, fill_value=0)

    # Scale numeric features
    scaled = scaler.transform(df)

    # Apply PCA
    reduced = pca.transform(scaled)

    # Combine with text features (if available)
    if text_features is not None:
        final_features = np.hstack([reduced, text_features.toarray()])
    else:
        final_features = reduced

    return final_features

# ================================
# 3. Streamlit UI
# ================================
st.title("🧠 Mental Health Prediction App")
st.write("Fill in the details below and press **Predict** to see results from multiple models.")

# -------------------------------
# Demographics
# -------------------------------
st.header("📌 Demographics")
age = st.number_input("Age", min_value=10, max_value=100, value=25)
gender = st.selectbox("Gender", ["Male", "Female"])
current_place = st.text_input("Current place")
previous_place = st.text_input("Previous place before war")
study_degree = st.text_input("Study Degree")
study_status = st.text_input("Study Status")
marital_status = st.text_input("Marital status of family (parents)")
job_before = st.text_input("Job before war")
job_current = st.text_input("Job currently")
living_with = st.text_input("Who do you currently live with?")
income_before = st.number_input("Income level before war", min_value=0, max_value=20, value=5)
income_after = st.number_input("Income level after war", min_value=0, max_value=20, value=3)

# -------------------------------
# War exposure
# -------------------------------
st.header("⚔️ War Exposure")
leave_home = st.selectbox("Leave home for war", ["yes", "no"])
witnessed_violence = st.selectbox("Witnessed direct violence", ["yes", "no"])
hurt_relatives = st.selectbox("Hurt or relatives harmed", ["yes", "no"])
lost_persons = st.selectbox("Lost persons", ["yes", "no"])
arrested = st.selectbox("You arrested", ["yes", "no"])
lack_food = st.selectbox("Lack of food or medicine", ["yes", "no"])
property_destroyed = st.selectbox("Your property destroyed", ["yes", "no"])
afraid_life = st.selectbox("Felt afraid for losing life", ["yes", "no"])

# -------------------------------
# Social / family support
# -------------------------------
st.header("👨‍👩‍👧 Social & Family Support")
feel_love = st.selectbox("Feel love", ["yes", "no"])
family_conflicts = st.selectbox("Hard disagreements in family", ["yes", "no"])
free_to_talk = st.selectbox("Feel free to talk", ["yes", "no"])
close_friends = st.selectbox("Have close friends", ["yes", "no"])
feel_alone = st.selectbox("Feel alone", ["yes", "no"])
feel_bullying = st.selectbox("Feel bullying", ["yes", "no"])
part_of_society = st.selectbox("Feel part of society", ["yes", "no"])

# -------------------------------
# Psychological symptoms
# -------------------------------
st.header("🧩 Psychological Symptoms")
not_interested = st.selectbox("Not interested to do things", ["yes", "no"])
frustrating = st.selectbox("Feel frustrating", ["yes", "no"])
sleeping_problems = st.selectbox("Sleeping problems", ["yes", "no"])
tired = st.selectbox("Feel tired", ["yes", "no"])
eating_problems = st.selectbox("Eating problems", ["yes", "no"])
focus_problems = st.selectbox("Focus problems", ["yes", "no"])
uncomfortable = st.selectbox("Feel uncomfortable", ["yes", "no"])
hurt_yourself = st.selectbox("Hurt yourself", ["yes", "no"])
stressed = st.selectbox("Feel stressed", ["yes", "no"])
difficulty_relaxing = st.selectbox("Difficulty relaxing", ["yes", "no"])
angry = st.selectbox("Feeling easily angry", ["yes", "no"])
fearful = st.selectbox("Feeling fearful", ["yes", "no"])

# PTSD-specific items
st.subheader("PTSD Indicators")
unwanted_memories = st.selectbox("Unwanted memories", ["yes", "no"])
disturbing_dreams = st.selectbox("Disturbing dreams", ["yes", "no"])
avoided_thoughts = st.selectbox("Avoided thinking/talking", ["yes", "no"])
ignore_reminders = st.selectbox("Ignore reminders", ["yes", "no"])
lost_interest = st.selectbox("Lost interest", ["yes", "no"])
irritable = st.selectbox("Feel irritable", ["yes", "no"])
overly_alert = st.selectbox("Overly alert", ["yes", "no"])
concentration = st.selectbox("Difficulty concentrating", ["yes", "no"])
sleeping_diff = st.selectbox("Difficulty sleeping", ["yes", "no"])

# -------------------------------
# Personality disorder items
# -------------------------------
st.header("🌀 Personality Traits")
doubt_loyalty = st.selectbox("Doubt loyalty", ["yes", "no"])
difficult_forgive = st.selectbox("Difficult to forgive", ["yes", "no"])
not_share_secrets = st.selectbox("Not share secrets", ["yes", "no"])
react_strongly = st.selectbox("React strongly", ["yes", "no"])
no_social_enjoyment = st.selectbox("Don't enjoy social relationships", ["yes", "no"])
indifferent = st.selectbox("Emotionally indifferent", ["yes", "no"])
impulsive = st.selectbox("Act impulsively without thinking", ["yes", "no"])
difficult_promises = st.selectbox("Difficult to keep promises", ["yes", "no"])
may_lie = st.selectbox("May lie sometimes", ["yes", "no"])
get_into_fights = st.selectbox("Get into fights", ["yes", "no"])
difficult_rules = st.selectbox("Difficult to follow rules", ["yes", "no"])

# -------------------------------
# Needs (TF-IDF text)
# -------------------------------
st.header("📑 Needs & Support (Free Text)")
needs_text = st.text_area("Describe your needs (emotional, financial, psychological, etc.)")

# -------------------------------
# Scores
# -------------------------------
st.header("📊 Scores")
total_symptoms = st.number_input("Total symptoms score", min_value=0, max_value=100, value=10)
trauma_score = st.number_input("Trauma exposure score", min_value=0, max_value=20, value=2)
social_support_score = st.number_input("Social support score", min_value=0, max_value=20, value=5)

# -------------------------------
# Build input dictionary
# -------------------------------
user_input = {
    "Age": age,
    "Gender": gender,
    "Current place": current_place,
    "previous place before war": previous_place,
    "Study Degree": study_degree,
    "study status": study_status,
    "Marital status of family (parents)": marital_status,
    "job before war": job_before,
    "job currently": job_current,
    "Who do you currently live with?": living_with,
    "income level before war": income_before,
    "income level after war": income_after,
    "leave home for war": leave_home,
    "witnessed direct violence": witnessed_violence,
    "hurt or relatives hard": hurt_relatives,
    "lost persons": lost_persons,
    "you arrested": arrested,
    "Lack of food or medicine": lack_food,
    "Your property destroyed": property_destroyed,
    "felt afraid for lossing life": afraid_life,
    "Feel love": feel_love,
    "Hard disagreements in family": family_conflicts,
    "feel free to talk": free_to_talk,
    "close friends": close_friends,
    "feel alone": feel_alone,
    "feel Bullying": feel_bullying,
    "fell part of society": part_of_society,
    "not interested to do sth": not_interested,
    "feel frustrating": frustrating,
    "sleeping problems": sleeping_problems,
    "feel tired": tired,
    "eating problems": eating_problems,
    "focus problems": focus_problems,
    "feel Uncomfortable": uncomfortable,
    "hrut yourself": hurt_yourself,
    "feel stressed": stressed,
    "Difficulty relaxing": difficulty_relaxing,
    "Feeling easily angry": angry,
    "Feeling fearful": fearful,
    "unwant memories": unwanted_memories,
    "disturbing dreams": disturbing_dreams,
    "avoided thinking or talking": avoided_thoughts,
    "ignore reminders": ignore_reminders,
    "lost interest": lost_interest,
    "feel irritable": irritable,
    "overly alert": overly_alert,
    "difficulty concentrating": concentration,
    "difficulty sleeping": sleeping_diff,
    "doubt loyalty": doubt_loyalty,
    "difficult to forgive": difficult_forgive,
    "not share secrets": not_share_secrets,
    "react strongly": react_strongly,
    "don't enjoy social relationships": no_social_enjoyment,
    "emotionally indifferent": indifferent,
    "impulsively without thinking": impulsive,
    "find difficult keep promises": difficult_promises,
    "may lie sometimes": may_lie,
    "get into fights": get_into_fights,
    "difficult follow rules": difficult_rules,
    "needs_text": needs_text,
    "total_symptoms_score": total_symptoms,
    "trauma_exposure_score": trauma_score,
    "social_support_score": social_support_score,
}

# ================================
# 4. Run Predictions
# ================================
if st.button("Predict"):
    features = preprocess_input(user_input)

    st.subheader("🔮 Model Predictions")
    results = {}
    for disorder, model_dict in models.items():
        results[disorder] = {}
        for model_name, model in model_dict.items():
            pred = model.predict(features)[0]
            if hasattr(model, "predict_proba"):
                prob = np.max(model.predict_proba(features))
                results[disorder][model_name] = f"{pred} ({prob:.2f})"
            else:
                results[disorder][model_name] = str(pred)

    results_df = pd.DataFrame(results).T
    st.dataframe(results_df)

    st.success("✅ Predictions complete! Compare models above.")
