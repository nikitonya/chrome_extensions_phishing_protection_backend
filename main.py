from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer
from flask_cors import CORS
from transformers import AutoConfig, AutoModelForSequenceClassification

app = Flask(__name__)
CORS(app)

model_path = "ealvaradob/bert-finetuned-phishing"
config = AutoConfig.from_pretrained(model_path)
config.num_labels = 2
config.problem_type = "single_label_classification"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config)

def predict_url(url):
    inputs = tokenizer(url, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits).item()
    return predicted_class

@app.route('/')
def welcome():
    return "Hello"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        url = request.json['url']
        print(f"URL: {url}")
        predicted_class = predict_url(url)
        print(f"Предсказанный класс: {predicted_class}")
        return jsonify({"predicted_class": predicted_class})
    except Exception as e:
        print(f"Ошибка: {str(e)}")
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
