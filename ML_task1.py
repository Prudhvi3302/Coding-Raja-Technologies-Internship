from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
nltk.download('stopwords')
nltk.download('punkt')
data = {
    'text': [
        "I love this product!",
        "Terrible experience, would not recommend.",
        "Neutral comment about the service."
    ],
    'sentiment': ['positive', 'negative', 'neutral']
}
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)
data['text'] = [preprocess_text(text) for text in data['text']]
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['sentiment'], test_size=0.2, random_state=42)
model = make_pipeline(TfidfVectorizer(), SVC())
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict_sentiment():
    data = request.get_json(force=True)
    text = data['text']
    processed_text = preprocess_text(text)
    prediction = model.predict([processed_text])[0]
    return jsonify({'sentiment': prediction})
if __name__ == '__main__':
    app.run(debug=True)
