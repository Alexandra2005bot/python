import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss


#подготовка данных
features = ['age', 'weight', 'height', 'ap_lo', 'ap_hi']
target = 'cardio'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)



knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

#Байес
nb = GaussianNB()
nb.fit(X_train, y_train)


dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)


models = {
    'kNN': knn,
    'Naive Bayes': nb,
    'Decision Tree': dt
}

results = {}

for name, model in models.items():
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    loss = log_loss(y_test, y_proba)

    results[name] = {
        'Accuracy': acc,
        'F1-score': f1,
        'ROC AUC': roc_auc,
        'Log Loss': loss
    }



df_results = pd.DataFrame(results).T
print(df_results)

# Определение лучших по каждой метрике
best_metrics = df_results.idxmax()
print("\nЛучший классификатор по каждой метрике:")
print(best_metrics)