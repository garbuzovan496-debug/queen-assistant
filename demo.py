import torch
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("AI Ticket Classifier")
print("="*50)

print("Загрузка модели...")
model = AutoModelForSequenceClassification.from_pretrained("./model")
tokenizer = AutoTokenizer.from_pretrained("./model")

with open("./model/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"Модель загружена на {device}")
print(f"Категорий: {len(le.classes_)}")

def classify(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        pred = outputs.logits.argmax(-1).item()
    return le.inverse_transform([pred])[0]

print("Введите текст заявки (exit - выход):")
while True:
    text = input("Текст: ")
    if text.lower() in ["exit", "quit", "q"]:
        print("До свидания!")
        break
    if not text.strip():
        continue
    result = classify(text)
    print(f"Категория: {result}")
