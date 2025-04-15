
import pandas as pd
import numpy as np
from scipy.spatial import distance
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier  # Для сравнения с реализацией sklearn

# 1. Загрузка данных

data = pd.read_csv("mlbootcamp5_train.csv")

# 2. Подготовка данных

X = data[['age', 'weight', 'height', 'ap_lo', 'ap_hi']].values  # Выбранные признаки
y = data['cardio'].values

# 3. Реализация kNN классификатора (с комментариями)

def knn_classify(X_train, y_train, X_test, k, metric='euclidean'):
    """
    Классификатор k-ближайших соседей.

    Args:
        X_train: Обучающая выборка (признаки).
        y_train: Метки классов обучающей выборки.
        X_test: Тестовая выборка (признаки).
        k: Количество ближайших соседей.
        metric: Метрика расстояния ('euclidean', 'manhattan' и т.д.).

    Returns:
        predictions: Предсказанные метки классов для тестовой выборки.
    """
    predictions = []
    for test_point in X_test:
        # 1. Считаем расстояния до всех точек в обучающей выборке
        distances = [distance.cdist([test_point], [train_point], metric)[0][0] for train_point in X_train]  # distance.cdist эффективнее
        # 2. Сортируем индексы по расстоянию (argsort возвращает индексы)
        nearest_neighbor_ids = np.argsort(distances)[:k]

        # 3. Отбираем метки классов ближайших соседей
        nearest_neighbor_labels = y_train[nearest_neighbor_ids]

        # 4. Голосование (находим наиболее часто встречающуюся метку)
        #   np.bincount считает количество вхождений каждого значения
        #   np.argmax возвращает индекс максимального значения (т.е. наиболее частую метку)
        prediction = np.argmax(np.bincount(nearest_neighbor_labels))
        predictions.append(prediction)

    return np.array(predictions)


# 4. Кросс-валидация для поиска оптимального k

def evaluate_knn(X, y, k_values, cv=5, metric='euclidean'):
    """
    Оценивает производительность kNN классификатора с помощью кросс-валидации.

    Args:
        X: Признаки.
        y: Метки классов.
        k_values: Список значений k для проверки.
        cv: Количество фолдов в кросс-валидации.
        metric: Метрика расстояния.

    Returns:
        results: Словарь, содержащий результаты кросс-валидации для каждого k.
    """
    results = {}
    for k in k_values:
        accuracy_scores = []
        f1_scores = []
        roc_auc_scores = []
        log_loss_scores = []

        # Создаем объект StratifiedKFold для стратифицированной кросс-валидации
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42) # Важно для сохранения баланса классов
        
        for train_index, test_index in skf.split(X, y):  # split возвращает индексы для train и test
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Обучаем и предсказываем с помощью kNN
            predictions = knn_classify(X_train, y_train, X_test, k, metric)

            # Рассчитываем метрики
            accuracy = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions)
            roc_auc = roc_auc_score(y_test, predictions)
            logloss = log_loss(y_test, predictions)  # Log Loss

            accuracy_scores.append(accuracy)
            f1_scores.append(f1)
            roc_auc_scores.append(roc_auc)
            log_loss_scores.append(logloss)

        # Сохраняем средние значения метрик для данного k
        results[k] = {
            'accuracy': np.mean(accuracy_scores),
            'f1': np.mean(f1_scores),
            'roc_auc': np.mean(roc_auc_scores),
            'log_loss': np.mean(log_loss_scores)
        }

    return results


# 5. Определяем параметры для kNN, Naive Bayes и Decision Tree.
k_values = [3, 5, 7, 9, 11, 13, 15]  # Диапазон значений k для kNN
cv = 5  # Количество фолдов для кросс-валидации

# Оцениваем kNN
knn_results = evaluate_knn(X, y, k_values, cv)
print("kNN Results:\n", knn_results)


# 6. Обучение и оценка Naive Bayes

gnb = GaussianNB()
nb_accuracy_scores = []
nb_f1_scores = []
nb_roc_auc_scores = []
nb_log_loss_scores = []

skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    gnb.fit(X_train, y_train)  # Обучаем Naive Bayes
    predictions = gnb.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, predictions)
    logloss = log_loss(y_test, predictions)

    nb_accuracy_scores.append(accuracy)
    nb_f1_scores.append(f1)
    nb_roc_auc_scores.append(roc_auc)
    nb_log_loss_scores.append(logloss)

nb_results = {
    'accuracy': np.mean(nb_accuracy_scores),
    'f1': np.mean(nb_f1_scores),
    'roc_auc': np.mean(nb_roc_auc_scores),
    'log_loss': np.mean(nb_log_loss_scores)
}
print("Naive Bayes Results:\n", nb_results)

# 7. Обучение и оценка Decision Tree

dtc = DecisionTreeClassifier(random_state=42)
dt_accuracy_scores = []
dt_f1_scores = []
dt_roc_auc_scores = []
dt_log_loss_scores = []

skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    dtc.fit(X_train, y_train)
    predictions = dtc.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, predictions)
    logloss = log_loss(y_test, predictions)

    dt_accuracy_scores.append(accuracy)
    dt_f1_scores.append(f1)
    dt_roc_auc_scores.append(roc_auc)
    dt_log_loss_scores.append(logloss)

dt_results = {
    'accuracy': np.mean(dt_accuracy_scores),
    'f1': np.mean(dt_f1_scores),
    'roc_auc': np.mean(dt_roc_auc_scores),
    'log_loss': np.mean(dt_log_loss_scores)
}
print("Decision Tree Results:\n", dt_results)

# 8. Сравнение результатов и вывод лучшего классификатора для каждой метрики
def find_best_classifier(knn_results, nb_results, dt_results):
    """
    Сравнивает результаты классификаторов и определяет лучшего для каждой метрики.
    """
    best_k = None
    best_accuracy = 0
    for k, scores in knn_results.items():
        if scores['accuracy'] > best_accuracy:
            best_accuracy = scores['accuracy']
            best_k = k

    print(f"Лучший kNN: k={best_k} с accuracy={best_accuracy}")

    results = {
        'accuracy': {'kNN': best_accuracy, 'Naive Bayes': nb_results['accuracy'], 'Decision Tree': dt_results['accuracy']},
        'f1': {'kNN': knn_results[best_k]['f1'], 'Naive Bayes': nb_results['f1'], 'Decision Tree': dt_results['f1']},
        'roc_auc': {'kNN': knn_results[best_k]['roc_auc'], 'Naive Bayes': nb_results['roc_auc'], 'Decision Tree': dt_results['roc_auc']},
        'log_loss': {'kNN': knn_results[best_k]['log_loss'], 'Naive Bayes': nb_results['log_loss'], 'Decision Tree': dt_results['log_loss']}
    }

    for metric, scores in results.items():
        best_classifier = max(scores, key=scores.get)  # max() ищет ключ с максимальным значением
        print(f"Лучший классификатор для {metric}: {best_classifier} с {metric}={scores[best_classifier]}")

find_best_classifier(knn_results, nb_results, dt_results)

# 9. Сравнение с реализацией sklearn (пример, можно убрать)

knn_sklearn = KNeighborsClassifier(n_neighbors=5)  # Пример использования sklearn
skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
accuracy_sklearn = []
for train_index, test_index in skf.split(X,y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    knn_sklearn.fit(X_train, y_train)
    predictions = knn_sklearn.predict(X_test)
    accuracy_sklearn.append(accuracy_score(y_test, predictions))

print(f"Accuracy sklearn kNN (k=5): {np.mean(accuracy_sklearn)}")
