# Customer churn prediction (coffee delivery)

Проект: прогноз оттока клиентов сервиса доставки кофе. Задача — бинарная классификация: предсказать вероятность того, что клиент уйдёт в следующем месяце.

## Goal
- Построить интерпретируемую модель, прогнозирующую `P(churn=1)` для каждого клиента.
- Оценить качество на несбалансированных данных.
- Подготовить решение к применению: пайплайн предобработки + модель, сохранение артефактов.

## Data
Витрина признаков за последние 4 недели (1 строка = 1 клиент), числовые и категориальные признаки.

Пример признаков:
- активность и покупки: `days_since_last_order`, `order_frequency_*`, `total_spent_*`
- предпочтения: `last_coffee_type`, `preferred_roast`, `milk_preference`, `coffee_bean_origin`
- взаимодействие с приложением: `app_opens_per_week`, `notifications_enabled`, `app_crashes_last_month`
- целевая: `churn` (0/1)

> Данные не хранятся в репозитории. Для запуска положите файл в `data/coffee_churn_dataset.csv`
> или используйте ваш локальный путь/источник данных.

## Approach
- Baseline: `DummyClassifier`
- Основная модель: `LogisticRegression` (интерпретируемая)
- Оценка: k-fold cross-validation
- Подбор гиперпараметров (при необходимости): `GridSearchCV`
- Полный пайплайн без утечек: предобработка + модель в `sklearn.Pipeline`
- Сохранение артефактов: `joblib/pickle`

## Metrics
- Основная: **PR-AUC**
- Дополнительно: ROC-AUC, Precision/Recall/F1, LogLoss (если применялось)

## Project structure
- `coffee_churn_prediction.ipynb` — основной ноутбук
- `requirements.txt` — зависимости
- `data/` — папка для датасета (не хранится в репозитории)

## How to run
1. Установите зависимости:
   ```bash
   pip install -r requirements.txt
