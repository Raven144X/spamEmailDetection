import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

@st.cache_resource
def load_data_and_train_model():
    url = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv"

    df = pd.read_csv(url,sep="\t",header=None,names=["label","messages"])
    df["label_num"] = df["label"].map({"ham":0,"spam":1})

    x = df["messages"]
    y = df["label_num"]

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

    vectorizer = CountVectorizer(stop_words="english")
    x_train_transformed = vectorizer.fit_transform(x_train)
    x_test_transformed = vectorizer.transform(x_test)

    model = LogisticRegression()
    model.fit(x_train_transformed, y_train)

    accuracy = model.score(x_test_transformed, y_test)
    total_samples = len(df)

    return vectorizer, model, accuracy, total_samples


with st.spinner("Downloading dataset and training Logistic Regression..."):
    vectorizer, model, accuracy, total_samples = load_data_and_train_model()

st.title("Spam email detection")
st.write(
    "Paste the contents of an email below, and our Logistic Regression model "
    "will calculate the probability of it being spam."
)


with st.sidebar:
    st.header("Model Statistics")
    st.metric(label="Dataset Size", value=f"{total_samples} messages")
    st.metric(label="Testing Accuracy", value=f"{accuracy * 100:.2f}%")
    st.markdown(
        """
    **Dataset Details:**
    Uses the UCI SMS Spam Collection dataset consisting of real mobile phone notifications and emails.
    """
    )


st.markdown("---")


def clear_text():
    st.session_state["text_box"] = ""


user_text = st.text_area(label="Enter email text",
                         placeholder="Type or Paste your email here",
                         height=150,
                         key="text_box")
col_bt1,col_bt2 = st.columns([1,1])


with col_bt1:
    predict = st.button("Predict",use_container_width=True)
with col_bt2:
    delete = st.button("🗑️",on_click=clear_text,use_container_width=False)






if predict:
    if user_text.strip() == "":
        st.warning("Write something BIG brain")
    else:
        transformed_input = vectorizer.transform([user_text])

        prediction = model.predict(transformed_input)[0]
        spam_probability = model.predict_proba(transformed_input)[0][1]

        st.markdown("### 🔎 Scan Diagnostics")

        if prediction == 1:
            st.error(
                "🚨 **CRITICAL WARNING:** This text exhibits high-probability patterns of a phishing or scam attempt.")
        else:
            st.success("✅ **SECURE:** This text appears safe and conforms to standard communicative language.")

        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="Spam Confidence Score",
                value=f"{spam_probability * 100:.1f}%",
            )
        with col2:
            classification_type = "SPAM" if prediction == 1 else "HAM"
            st.metric(label="Assigned Class Output", value=classification_type)