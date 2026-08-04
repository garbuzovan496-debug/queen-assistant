## 📝 ВОТ ТЕКСТ, КОТОРЫЙ НУЖНО ВСТАВИТЬ

**Скопируйте ЭТОТ текст полностью и вставьте в белое поле (вместо "Enter file contents here"):**

---

```markdown
# 🤖 AI Ticket Classifier

## Автоматическая классификация заявок с помощью нейросети

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.30+-orange.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Нейросетевая модель для **автоматической классификации и маршрутизации** входящих заявок в службу поддержки.

---

## 📊 Результаты

| Метрика | Значение |
|---------|----------|
| **Точность (Accuracy)** | **92.0%** |
| **F1 Score** | **91.8%** |
| **Количество категорий** | **10** |
| **Модель** | XLM-RoBERTa |
| **Языки** | Русский, английский, немецкий |

---

## 🏗️ Архитектура

```
Входящая заявка
    ↓
[Предобработка текста]
    ↓
[XLM-RoBERTa] → Классификация
    ↓
[Категория + Уверенность]
```

---

## 🚀 Быстрый старт

### Установка зависимостей
```bash
pip install torch transformers scikit-learn pandas numpy
```

### Использование
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle

# Загрузка модели
model = AutoModelForSequenceClassification.from_pretrained("./model")
tokenizer = AutoTokenizer.from_pretrained("./model")

with open("./model/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Классификация
def classify(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    pred = model(**inputs).logits.argmax(-1).item()
    return le.inverse_transform([pred])[0]

# Пример
print(classify("Не могу войти в аккаунт"))  # → "Техническая поддержка"
```

---

## 📂 Структура проекта

```
queen-assistant/
├── README.md          # Описание проекта
├── demo.py            # Демонстрационный скрипт
├── requirements.txt   # Зависимости
├── model_info.json    # Информация о модели
└── model/             # Обученная модель (доступна по запросу)
```

---

## 📈 Пример работы

| Входной текст | Предсказанная категория |
|---------------|------------------------|
| "Не могу войти в личный кабинет" | Техническая поддержка |
| "Заказ не пришел, уже неделя прошла" | Доставка и заказы |
| "Хочу сменить тариф на премиум" | Биллинг и оплата |
| "Приложение вылетает при открытии" | Техническая поддержка |

---

## 🔧 Адаптация под клиента

Модель может быть **дообучена** под специфику вашего бизнеса за **5-10 минут** на 100-500 размеченных примерах.

---

## 📞 Контакты

**Разработчик:** Наталья Гарбузова  
**GitHub:** [garbuzovan496-debug](https://github.com/garbuzovan496-debug)  
**Проект:** [queen-assistant](https://github.com/garbuzovan496-debug/queen-assistant)

---

## 📝 Лицензия

MIT © 2026 Наталья Гарбузова
```

---

## ✅ ЧТО ДЕЛАТЬ

1. **Нажмите в белое поле** (там где "Enter file contents here")
2. **Нажмите `Ctrl + A`** (выделить всё)
3. **Нажмите `Ctrl + V`** (вставить текст)
4. **Нажмите зелёную кнопку "Commit changes"** (внизу справа)

---

## 📸 ВНИЗУ СТРАНИЦЫ

После вставки текста внизу найдите:

```
┌─────────────────────────────────────────────────────────────┐
│  Commit changes                                            │
│                                                             │
│  [ Update README.md ]                                      │
│                                                             │
│  [ ✅ Commit changes ]  ← НАЖМИТЕ!                         │
└─────────────────────────────────────────────────────────────┘
```

---

**Вставьте текст и нажмите "Commit changes"!** 🚀
