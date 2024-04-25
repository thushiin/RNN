from flask import Flask, render_template, request
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load the model
mod = joblib.load('fakenews.joblib')

ps = PorterStemmer()

def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [ps.stem(t) for t in text if t not in stopwords.words('english')]
    return ' '.join(text)

def encode_text(text, vocab_size=5000, sent_len=20):
    text = preprocess_text(text)
    encoded = [one_hot(text, vocab_size)]
    encoded = pad_sequences(encoded, maxlen=sent_len)
    return encoded

@app.route('/')
def home():
    return render_template('fakenews_detect.html')

@app.route('/predict', methods=['POST'])
def pred():
    if request.method == 'POST':
        message = request.form['message']
        xdata = encode_text(message)
        prediction = mod.predict(xdata)
        if prediction<0.5:
            predict='FAKE'
        else:
            predict='REAL'
        return render_template('fakenews_detect.html', prediction=predict)

if __name__ == '__main__':
    app.run(debug=True)
