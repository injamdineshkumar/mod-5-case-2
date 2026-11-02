
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "spam_model.pkl"
DATA_PATH = BASE / "spam.csv"

st.set_page_config(
    page_title="Spam Detection",
    page_icon="📩",
    layout="wide"
)

st.title("📩 Spam vs Non-Spam Classifier")
st.write("Machine Learning Project using Python and Streamlit")

# Dataset preparation
def prepare_data(df):
    columns = {str(c).lower().strip(): c for c in df.columns}

    if "label" in columns and "message" in columns:
        label_col = columns["label"]
        message_col = columns["message"]
    elif "v1" in columns and "v2" in columns:
        label_col = columns["v1"]
        message_col = columns["v2"]
    else:
        raise ValueError("CSV needs label/message or v1/v2 columns.")

    df = df[[label_col, message_col]].copy()
    df.columns = ["label", "message"]
    df = df.dropna()

    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df["message"] = df["message"].astype(str).str.strip()

    df["label"] = df["label"].replace({
        "non-spam": "ham",
        "nonspam": "ham"
    })

    df = df[
        df["label"].isin(["spam", "ham"]) &
        (df["message"] != "")
    ]

    if df["label"].nunique() != 2:
        raise ValueError("Both spam and ham classes are required.")

    return df


# Sidebar navigation
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    ["Home", "Train Model", "Predict Message"]
)

uploaded = st.sidebar.file_uploader(
    "Upload SMS CSV",
    type=["csv"]
)

# Load dataset
data = None

try:
    if uploaded is not None:
        data = prepare_data(
            pd.read_csv(uploaded, encoding="latin-1")
        )
    elif DATA_PATH.exists():
        data = prepare_data(
            pd.read_csv(DATA_PATH, encoding="latin-1")
        )
except Exception as e:
    st.error(f"Dataset error: {e}")


# HOME PAGE
if page == "Home":

    st.header("Welcome to Spam Detection System")

    col1, col2, col3 = st.columns(3)

    col1.metric("Classification", "Binary")
    col2.metric("Feature Extraction", "TF-IDF")
    col3.metric("Feature Selection", "Chi-square")

    st.subheader("Project Description")

    st.write("""
    This project detects spam and non-spam SMS messages
    using machine learning algorithms.

    Techniques used:
    - TF-IDF
    - Chi-square feature selection
    - Multinomial Naive Bayes
    """)

    if data is not None:
        st.subheader("Dataset Preview")
        st.dataframe(data.head(10), use_container_width=True)

        st.subheader("Message Distribution")

        counts = data["label"].value_counts().rename(
            index={"ham": "Non-Spam", "spam": "Spam"}
        )

        st.bar_chart(counts)

        st.success(f"Total messages: {len(data)}")
    else:
        st.warning("Upload a CSV dataset or add spam.csv.")


# TRAIN MODEL PAGE
elif page == "Train Model":

    st.header("Train Machine Learning Model")

    if data is None:
        st.warning("Please load a dataset first.")

    else:
        test_size = st.slider(
            "Test data percentage",
            10, 40, 20, 5
        )

        if st.button("Train Model", type="primary"):

            X_train, X_test, y_train, y_test = train_test_split(
                data["message"],
                data["label"],
                test_size=test_size / 100,
                random_state=42,
                stratify=data["label"]
            )

            # Find the available number of features
            probe = TfidfVectorizer(
                ngram_range=(1, 2),
                sublinear_tf=True
            )

            feature_count = probe.fit_transform(X_train).shape[1]
            k = min(3000, feature_count)

            # ML pipeline
            model = Pipeline([
                ("tfidf", TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    sublinear_tf=True
                )),
                ("chi2", SelectKBest(
                    score_func=chi2,
                    k=k
                )),
                ("classifier", MultinomialNB(alpha=0.5))
            ])

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            # Performance evaluation
            accuracy = accuracy_score(y_test, predictions)

            precision = precision_score(
                y_test, predictions,
                pos_label="spam", zero_division=0
            )

            recall = recall_score(
                y_test, predictions,
                pos_label="spam", zero_division=0
            )

            f1 = f1_score(
                y_test, predictions,
                pos_label="spam", zero_division=0
            )

            # Save model
            joblib.dump(model, MODEL_PATH)
            st.session_state["spam_model"] = model

            st.success("Model trained successfully!")

            st.subheader("Performance Results")

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Accuracy", f"{accuracy:.2%}")
            c2.metric("Precision", f"{precision:.2%}")
            c3.metric("Recall", f"{recall:.2%}")
            c4.metric("F1 Score", f"{f1:.2%}")

            st.subheader("Confusion Matrix")

            cm = confusion_matrix(
                y_test, predictions,
                labels=["ham", "spam"]
            )

            st.dataframe(pd.DataFrame(
                cm,
                index=["Actual Non-Spam", "Actual Spam"],
                columns=["Predicted Non-Spam", "Predicted Spam"]
            ))

            st.subheader("Classification Report")

            st.text(classification_report(
                y_test, predictions,
                zero_division=0
            ))

            st.info("Model saved as spam_model.pkl")


# PREDICTION PAGE
elif page == "Predict Message":

    st.header("Spam Message Detection")

    message = st.text_area(
        "Enter your SMS message",
        placeholder="Congratulations! You won a free prize."
    )

    if st.button("Check Message", type="primary"):

        model = st.session_state.get("spam_model")

        if model is None and MODEL_PATH.exists():
            model = joblib.load(MODEL_PATH)

        if not message.strip():
            st.warning("Please enter a message.")

        elif model is None:
            st.warning("Train the model first.")

        else:
            prediction = model.predict([message])[0]

            probabilities = model.predict_proba([message])[0]
            classes = list(model.classes_)

            confidence = probabilities[
                classes.index(prediction)
            ]

            if prediction == "spam":
                st.error("🚨 SPAM MESSAGE DETECTED!")
            else:
                st.success("✅ NON-SPAM MESSAGE!")

            st.write(
                f"Predicted-class probability: {confidence:.2%}"
            )
