import torch
from datasets import load_dataset, Dataset
import os
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

new_model = "/home/user/github/chat-bot/SA-v2"
new_model_half = "/home/user/github/SA-v2-half-neutral-data"
base_model = "michellejieli/emotion_text_classifier"
tokenizer = AutoTokenizer.from_pretrained(base_model)

def preprocessing(data):
    data = data.rename_column("utterance", "text")
    data = data.rename_column("emotion", "label")
    data = data.remove_columns(["dialog_id", "turn_type"])
    return data

def predict(row):
    text = row['text']
    true_label = row['label']
    predicted_result = classifier(text)[0]
    predicted_label = label2id[predicted_result["label"]]

    return {"predicted_label": predicted_label, "true_label": true_label}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_labels = 7
id2label = {
    0: "neutral",
    1: "anger",
    2: "disgust",
    3: "fear",
    4: "happiness",
    5: "sadness",
    6: "surprise"
}

label2id = {
    "neutral": 0,
    "anger": 1,
    "disgust": 2,
    "fear": 3,
    "happiness": 4,
    "sadness": 5,
    "surprise": 6
}

og_label2id = {
    "neutral": 0,
    "anger": 1,
    "disgust": 2,
    "fear": 3,
    "joy": 4,
    "sadness": 5,
    "surprise": 6
}

data_name = "benjaminbeilharz/better_daily_dialog"
data = load_dataset(data_name, split='test', num_proc=8)
data = preprocessing(data)
# print(data[0:5])
classifier_model = AutoModelForSequenceClassification.from_pretrained(new_model, num_labels=num_labels, id2label=id2label, label2id=label2id)
classifier = pipeline("sentiment-analysis", model=classifier_model, tokenizer=tokenizer, device=0)

predictions = data.map(predict)
true_labels = [p["true_label"] for p in predictions]
predicted_labels = [p["predicted_label"] for p in predictions]
ft_10 = predicted_labels[0:9]

f1_ft = f1_score(true_labels, predicted_labels, average='weighted')
accuracy_ft = accuracy_score(true_labels, predicted_labels)

classifier_model = AutoModelForSequenceClassification.from_pretrained(new_model_half, num_labels=num_labels, id2label=id2label, label2id=label2id)
classifier = pipeline("sentiment-analysis", model=classifier_model, tokenizer=tokenizer, device=0)

predictions = data.map(predict)
true_labels = [p["true_label"] for p in predictions]
predicted_labels = [p["predicted_label"] for p in predictions]
ft_half_10 = predicted_labels[0:9]

f1_ft_half = f1_score(true_labels, predicted_labels, average='weighted')
accuracy_ft_half = accuracy_score(true_labels, predicted_labels)

classifier_model = AutoModelForSequenceClassification.from_pretrained(base_model)
classifier = pipeline("sentiment-analysis", model=classifier_model, tokenizer=tokenizer, device=0)
def predict_origin(row):
    text = row['text']
    true_label = row['label']
    predicted_result = classifier(text)[0]
    predicted_label = og_label2id[predicted_result["label"]]
    return {"predicted_label": predicted_label, "true_label": true_label}

predictions = data.map(predict_origin)
true_labels = [p["true_label"] for p in predictions]
predicted_labels = [p["predicted_label"] for p in predictions]
og_10 = predicted_labels[0:9]
f1 = f1_score(true_labels, predicted_labels, average='weighted')
accuracy = accuracy_score(true_labels, predicted_labels)

print("Fine-tuned:")
print("F1-score:", f1_ft, )
print("Accuracy:", accuracy_ft)
print("\nFine-tuned-half:")
print("F1-score:", f1_ft_half, )
print("Accuracy:", accuracy_ft_half)
print("\nOriginal:")
print("F1-score:", f1, )
print("Accuracy:", accuracy)
print("\ntrue:", true_labels[0:10], "\nfine-tuned:", ft_10, "\nfine-tuned-half:", ft_half_10, "\noriginal:", og_10)