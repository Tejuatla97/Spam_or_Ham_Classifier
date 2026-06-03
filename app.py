import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="AI Spam Detector",
    page_icon="🚀",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>

/* Main Background */
.stApp {
    background: linear-gradient(
        180deg,
        #020b2d 0%,
        #03154d 40%,
        #020b2d 100%
    );
}

/* Remove Streamlit Menu */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Title */
.title {
    text-align:center;
    font-size:85px;
    font-weight:900;
    margin-bottom:5px;

    background: linear-gradient(
        90deg,
        #FFD700,
        #00E5FF,
        #FF4DFF
    );

    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

/* Subtitle */
.subtitle {
    text-align:center;
    color:white;
    font-size:28px;
    font-weight:bold;
    margin-bottom:30px;
}

/* Main Card */
.glass {
    background:#071a63;
    border:1px solid #2d5bff;
    border-radius:25px;
    padding:25px;
    box-shadow:0px 0px 20px rgba(0,150,255,0.25);
}

/* Label */
.stTextArea label {
    color:white !important;
    font-size:28px !important;
    font-weight:bold !important;
}

/* Text Area */
.stTextArea textarea {

    background:#f2f2f2 !important;
    color:#555 !important;

    font-size:24px !important;
    font-weight:600 !important;

    border-radius:25px !important;
    border:2px solid #ff4d88 !important;

    min-height:250px !important;
}

/* Placeholder */
.stTextArea textarea::placeholder {
    font-size:24px !important;
    color:#777 !important;
}

/* Button */
.stButton > button {

    width:100%;
    height:75px;

    border:none;
    border-radius:20px;

    font-size:30px;
    font-weight:bold;
    color:white;

    background: linear-gradient(
        90deg,
        #ff3b6b,
        #ffb300,
        #8cff66,
        #00d4ff,
        #d84cff
    );

    transition:0.3s;
}

.stButton > button:hover {
    transform:scale(1.02);
}

/* Spam Card */
.spam-card {
    background:#0a1d6e;
    border:2px solid #ff4d88;
    border-radius:20px;
    padding:25px;
    text-align:center;
    color:white;
    font-size:32px;
    font-weight:bold;
}

/* Ham Card */
.ham-card {
    background:#0a1d6e;
    border:2px solid #3dff6f;
    border-radius:20px;
    padding:25px;
    text-align:center;
    color:white;
    font-size:32px;
    font-weight:bold;
}

/* Section Heading */
.section {
    color:white;
    font-size:34px;
    font-weight:bold;
    text-align:center;
    margin-top:20px;
    margin-bottom:20px;
}

/* Dataset Heading */
.dataset-title {
    color:white;
    font-size:36px;
    font-weight:bold;
}

/* Metric Styling */
[data-testid="stMetric"] {
    background:#071a63;
    border-radius:15px;
    padding:15px;
    border:1px solid #2d5bff;
}

[data-testid="stMetricLabel"] {
    color:white;
}

[data-testid="stMetricValue"] {
    color:white;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.markdown("""
<div class="title">
🚀 AI SPAM DETECTOR 🚀
</div>

<div class="subtitle">
Detect Spam & Ham Messages Using Machine Learning
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# DATASET
# --------------------------------------------------
data = {
    'message': [
        'Free free free money now',
        'Win cash prize',
        'Call me tonight',
        'Lets go for dinner',
        'Claim free reward now',
        'Are you attending class',
        'Free cash free offer',
        'Meet me tomorrow'
    ],
    'label': [
        'spam',
        'spam',
        'ham',
        'ham',
        'spam',
        'ham',
        'spam',
        'ham'
    ]
}

df = pd.DataFrame(data)

# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------
X = df["message"]
y = df["label"]

vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vectorized, y)

# --------------------------------------------------
# INPUT CARD
# --------------------------------------------------
st.markdown('<div class="glass">', unsafe_allow_html=True)

message = st.text_area(
    "📩 Enter Your Message",
    placeholder="Example: Win free cash reward now...",
    height=250
)

predict = st.button("🚀 Predict Message")

st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if predict:

    if message.strip() == "":
        st.warning("Please enter a message.")
    else:

        msg_vector = vectorizer.transform([message])

        prediction = model.predict(msg_vector)[0]
        probs = model.predict_proba(msg_vector)[0]

        spam_idx = list(model.classes_).index("spam")
        ham_idx = list(model.classes_).index("ham")

        spam_prob = probs[spam_idx]
        ham_prob = probs[ham_idx]

        st.markdown(
            '<div class="section">✨ Prediction Result ✨</div>',
            unsafe_allow_html=True
        )

        if prediction == "spam":
            st.markdown(
                '<div class="spam-card">🚨 SPAM MESSAGE 🚨</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="ham-card">✅ HAM MESSAGE ✅</div>',
                unsafe_allow_html=True
            )

        st.markdown(
            '<div class="section">📊 Prediction Confidence</div>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Spam Probability",
                f"{spam_prob:.2%}"
            )
            st.progress(float(spam_prob))

        with col2:
            st.metric(
                "Ham Probability",
                f"{ham_prob:.2%}"
            )
            st.progress(float(ham_prob))

# --------------------------------------------------
# DATASET TABLE
# --------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    '<div class="dataset-title">📋 Training Dataset</div>',
    unsafe_allow_html=True
)

st.dataframe(
    df,
    use_container_width=True
)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("""
<br><br>

<center>
<h3 style='color:white'>
✨ Built with Streamlit & Scikit-Learn ❤️
</h3>
</center>
""", unsafe_allow_html=True)