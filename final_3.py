import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report

# ---------- 1. Загрузка и подготовка данных ----------
df = pd.read_csv('Task 3 and 4_Loan_Data.csv')

feature_cols = [
    'credit_lines_outstanding',
    'loan_amt_outstanding',
    'total_debt_outstanding',
    'income',
    'years_employed',
    'fico_score',
]
target_col = 'default'

X = df[feature_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- 2. Модель 1: логистическая регрессия (интерпретируемая база) ----------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_scaled, y_train)

logreg_probs = logreg.predict_proba(X_test_scaled)[:, 1]
logreg_auc = roc_auc_score(y_test, logreg_probs)

# ---------- 3. Модель 2: случайный лес (нелинейные взаимодействия) ----------
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    min_samples_leaf=10,
    random_state=42,
)
rf.fit(X_train, y_train)

rf_probs = rf.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_probs)

# ---------- 4. Сравнительный анализ ----------
print(f"Logistic Regression AUC: {logreg_auc:.4f}")
print(f"Random Forest AUC:       {rf_auc:.4f}")
print()
print("Logistic Regression classification report:")
print(classification_report(y_test, logreg.predict(X_test_scaled)))
print("Random Forest classification report:")
print(classification_report(y_test, rf.predict(X_test)))

# Коэффициенты логрег — какие признаки сильнее всего влияют на PD
coef_df = pd.DataFrame({
    'feature': feature_cols,
    'coefficient': logreg.coef_[0],
}).sort_values('coefficient', key=abs, ascending=False)
print("\nВклад признаков (логрег, на стандартизованных данных):")
print(coef_df)

# Важность признаков — случайный лес
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf.importances_ if hasattr(rf, 'importances_') else rf.feature_importances_,
}).sort_values('importance', ascending=False)
print("\nВажность признаков (случайный лес):")
print(importance_df)

# ---------- 5. Финальная модель для продакшн-функции ----------
# Выбираем модель с лучшим AUC на тесте; для прототипа переобучаем на всех данных
best_model_name = 'random_forest' if rf_auc >= logreg_auc else 'logistic_regression'

if best_model_name == 'random_forest':
    final_model = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=10, random_state=42
    )
    final_model.fit(X[feature_cols], y)
    final_scaler = None
else:
    final_scaler = StandardScaler()
    X_scaled_full = final_scaler.fit_transform(X[feature_cols])
    final_model = LogisticRegression(max_iter=1000)
    final_model.fit(X_scaled_full, y)

print(f"\nВыбранная модель для оценки PD: {best_model_name}")


def predict_expected_loss(
    credit_lines_outstanding,
    loan_amt_outstanding,
    total_debt_outstanding,
    income,
    years_employed,
    fico_score,
    recovery_rate=0.10,
):
    """
    Возвращает вероятность дефолта (PD) и ожидаемый убыток (EL) по займу.

    EL = PD * (1 - recovery_rate) * EAD,
    где EAD (exposure at default) берём как остаток по займу loan_amt_outstanding —
    это сумма, которую кредитор реально теряет при дефолте по этому конкретному займу.
    """
    features = pd.DataFrame([{
        'credit_lines_outstanding': credit_lines_outstanding,
        'loan_amt_outstanding': loan_amt_outstanding,
        'total_debt_outstanding': total_debt_outstanding,
        'income': income,
        'years_employed': years_employed,
        'fico_score': fico_score,
    }])[feature_cols]

    if best_model_name == 'random_forest':
        pd_estimate = final_model.predict_proba(features)[0, 1]
    else:
        features_scaled = final_scaler.transform(features)
        pd_estimate = final_model.predict_proba(features_scaled)[0, 1]

    exposure_at_default = loan_amt_outstanding
    expected_loss = pd_estimate * (1 - recovery_rate) * exposure_at_default

    return {
        'probability_of_default': pd_estimate,
        'expected_loss': expected_loss,
    }


# ---------- 6. Тест на примерах ----------
test_cases = [
    dict(credit_lines_outstanding=0, loan_amt_outstanding=5221.545193,
         total_debt_outstanding=3915.471226, income=78039.38546,
         years_employed=5, fico_score=605),
    dict(credit_lines_outstanding=5, loan_amt_outstanding=1958.928726,
         total_debt_outstanding=8228.75252, income=26648.43525,
         years_employed=2, fico_score=572),
    dict(credit_lines_outstanding=7, loan_amt_outstanding=15000,
         total_debt_outstanding=25000, income=20000,
         years_employed=1, fico_score=520),  # заведомо рискованный профиль
]

for case in test_cases:
    result = predict_expected_loss(**case)
    print(f"\nВходные данные: {case}")
    print(f"  PD = {result['probability_of_default']:.4f}, "
          f"Expected Loss = {result['expected_loss']:.2f}")