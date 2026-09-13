# 🤖 AI Ticket Classifier

Автоматическая классификация и маршрутизация входящих заявок с помощью нейросети.

## 📊 Результаты

| Метрика | Значение |
|---|---|
| Точность (Accuracy) | **92.0%** |
| F1 Score | **91.8%** |
| Количество категорий | **10** |
| Модель | XLM-RoBERTa |
| Языки | Русский, английский, немецкий |

## 🏗️ Архитектура

Входящая заявка → предобработка → XLM-RoBERTa → категория + уверенность

## 🚀 Быстрый старт

Установка зависимостей:

    pip install torch transformers scikit-learn pandas numpy

Использование:

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import pickle

    model = AutoModelForSequenceClassification.from_pretrained("./model")
    tokenizer = AutoTokenizer.from_pretrained("./model")

    with open("./model/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    def classify(text):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
        pred = model(**inputs).logits.argmax(-1).item()
        return le.inverse_transform([pred])[0]

    print(classify("Не могу войти в аккаунт"))  # → "Техническая поддержка"

## 📂 Структура проекта

    queen-assistant/
    ├── README.md
    ├── AI_Ticket_Router.ipynb   # код обучения и инференса модели
    ├── demo.py                  # демонстрационный скрипт
    ├── requirements.txt
    ├── model_info.json
    └── model/                   # обученная модель (по запросу)

## 📈 Пример работы

| Входной текст | Предсказанная категория |
|---|---|
| Не могу войти в личный кабинет | Техническая поддержка |
| Заказ не пришёл, уже неделя прошла | Доставка и заказы |
| Хочу сменить тариф на премиум | Биллинг и оплата |
| Приложение вылетает при открытии | Техническая поддержка |

## 🔧 Адаптация под клиента

Модель дообучается под специфику бизнеса за 5–10 минут на 100–500 размеченных примерах.

## 📞 Контакты

**Разработчик:** Наталья Гарбузова
**GitHub:** [garbuzovan496-debug](https://github.com/garbuzovan496-debug)

## 📝 Лицензия

MIT © 2026 Наталья Гарбузова

