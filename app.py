# app.py - Attractive Fake News Detection UI
import streamlit as st
import torch
import re
import pickle
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

# ---------------------------
# Device
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Load trained model
# ---------------------------
model_path = r"C:\Users\Student\Documents\fakenewsprediction\data\fakenews_model.pkl"
vocab_path = r"C:\Users\Student\Documents\fakenewsprediction\data\vocab.pkl"

with open(model_path, "rb") as f:
    state_dict = pickle.load(f)

with open(vocab_path, "rb") as f:
    vocab = pickle.load(f)

# ---------------------------
# Model architecture
# ---------------------------
class FakeNewsModel(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.conv = torch.nn.Conv1d(embed_dim, 64, kernel_size=5)
        self.lstm = torch.nn.LSTM(64, hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, 2)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv(x))
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out

model = FakeNewsModel(vocab_size=len(vocab.word2idx)).to(device)
model.load_state_dict(state_dict)
model.eval()

stop_words = set(stopwords.words("english"))

# ---------------------------
# Prediction function
# ---------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

def predict_news(model, vocab, text, max_len=100):
    model.eval()
    text = clean_text(text)
    nums = [vocab.word2idx.get(word, 1) for word in text.split()]
    nums = nums[:max_len]
    if len(nums) < max_len:
        nums += [0]*(max_len - len(nums))
    input_tensor = torch.tensor([nums], dtype=torch.long).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
    return "Real" if predicted.item() == 1 else "Fake"

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(
    page_title="📰 Fake News Detection",
    page_icon="🗞️",
    layout="centered"
)

st.markdown("<h1 style='text-align: center; color: #4B0082;'>📰 Fake News Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #6A5ACD;'>Check if a news article is REAL or FAKE!</p>", unsafe_allow_html=True)

# Input area
news_input = st.text_area("Paste your news article here:", height=150, placeholder="Type or paste the news text...")

# Example buttons
st.markdown("### Try Example News Articles")
col1, col2 = st.columns(2)
with col1:
    if st.button("🌍 Real Example"):
        news_input = "The United Nations has released a report stating that global carbon emissions are expected to rise unless stronger climate actions are taken by major countries."
with col2:
    if st.button("⚠️ Fake Example"):
        news_input = "Scientists discovered that eating chocolate every day can make you live forever, study claims."

# Predict button
if st.button("Predict"):
    if news_input.strip() == "":
        st.warning("Please enter some text to predict!")
    else:
        prediction = predict_news(model, vocab, news_input)
        if prediction == "Real":
            st.success("✅ This news is likely REAL.")
        else:
            st.error("❌ This news is likely FAKE.")

# Footer
st.markdown("<hr style='border:1px solid #6A5ACD'>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Developed with ❤️ using Streamlit & PyTorch</p>", unsafe_allow_html=True)
