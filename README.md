# spamEmailDetection

# 🛡️ Spam Email Detection App

A lightweight, interactive web application that uses Machine Learning to analyze text patterns and predict the probability of a message being malicious spam or safe. Built completely in Python using **Streamlit** and **Scikit-Learn**.

Live Link: (https://spamemaildetection-7933o2tyxrx2oqrd2tkjqt.streamlit.app/)

---

## 🚀 Features
* **Real-time Machine Learning:** Leverages a trained **Logistic Regression** model to calculate exact classification probabilities ($P(y=1 \mid x; \theta)$).
* **Enterprise-Grade Dataset:** Trained on the classic **UCI SMS Spam Collection dataset** containing over 5,500 real-world messages with **~97.5% evaluation accuracy**.
* **Dynamic UI:** Includes an adaptive text interface with responsive layouts, action diagnostics, and a dynamic text-clearing feature via Streamlit session callbacks.
* **Side-by-Side Statistics Panel:** Embedded sidebar showing instantaneous model metrics (Dataset Size and Testing Accuracy) out of the box.

---

## 🛠️ Built With
* **Frontend UI:** [Streamlit](https://streamlit.io/)
* **Data Processing:** [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/)
* **Machine Learning Backbone:** [Scikit-Learn](https://scikit-learn.org/) (`CountVectorizer` + `LogisticRegression`)

---

## 📦 Local Installation & Setup

If you want to run this project locally on your machine, follow these steps:

### 1. Clone the Repository
```bash
git clone [https://github.com/Raven_144X/spamEmailDetection.git](https://github.com/Raven_144X/spamEmailDetection.git)
cd spamEmailDetection
```
### 2. Install Dependencies

```
pip install -r requirements.txt
```

### 3. Launch the application

```
streamlit run app.py
```


### Project Structure
├── app.py               
├── requirements.txt    
└── README.md            
