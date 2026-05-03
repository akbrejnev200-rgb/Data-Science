import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib, json, os, warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/mnt/user-data/uploads/loan_data.csv').drop('Loan_ID', axis=1)

# Imputation
df['Gender']           = df['Gender'].fillna(df['Gender'].mode()[0])
df['Married']          = df['Married'].fillna(df['Married'].mode()[0])
df['Self_Employed']    = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
df['Dependents']       = df['Dependents'].fillna(df['Dependents'].mode()[0])
df['Credit_History']   = df['Credit_History'].fillna(df['Credit_History'].mode()[0])
df['LoanAmount']       = df['LoanAmount'].fillna(df['LoanAmount'].median())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])

# Dependents
df['Dependents'] = df['Dependents'].replace('3+', '3')
df['Dependents'] = pd.to_numeric(df['Dependents'], errors='coerce').fillna(0).astype(int)

# Feature Engineering
df['TotalIncome']     = df['ApplicantIncome'] + df['CoapplicantIncome']
df['EMI_to_Income']   = (df['LoanAmount'] * 1000) / (df['Loan_Amount_Term'] * (df['TotalIncome'] + 1))
df['LoanAmount_log']  = np.log1p(df['LoanAmount'])
df['TotalIncome_log'] = np.log1p(df['TotalIncome'])
df['AppIncome_log']   = np.log1p(df['ApplicantIncome'])

# Encoding
df['Gender']        = df['Gender'].map({'Male': 1, 'Female': 0})
df['Married']       = df['Married'].map({'Yes': 1, 'No': 0})
df['Education']     = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
df['Self_Employed'] = df['Self_Employed'].map({'Yes': 1, 'No': 0})
df['Loan_Status']   = df['Loan_Status'].map({'Y': 1, 'N': 0})
df = pd.get_dummies(df, columns=['Property_Area'], drop_first=False)
df = df.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'TotalIncome'], axis=1)
df = df.dropna()

print(f"Dataset final: {df.shape}")
print(f"Colonnes: {df.columns.tolist()}")

X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# Logistic Regression
model_lr = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
model_lr.fit(X_train_sc, y_train)
y_pred_lr  = model_lr.predict(X_test_sc)
y_proba_lr = model_lr.predict_proba(X_test_sc)[:, 1]
cv_lr = cross_val_score(model_lr, X_train_sc, y_train, cv=5, scoring='roc_auc')
metrics_lr = {
    'accuracy': float(accuracy_score(y_test, y_pred_lr)),
    'precision': float(precision_score(y_test, y_pred_lr)),
    'recall': float(recall_score(y_test, y_pred_lr)),
    'f1': float(f1_score(y_test, y_pred_lr)),
    'auc': float(roc_auc_score(y_test, y_proba_lr)),
    'cv_auc_mean': float(cv_lr.mean()),
    'cv_auc_std': float(cv_lr.std()),
    'confusion_matrix': confusion_matrix(y_test, y_pred_lr).tolist(),
    'feature_importance': dict(zip(feature_names, np.abs(model_lr.coef_[0]).tolist()))
}

# Random Forest
model_rf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42, class_weight='balanced', n_jobs=-1)
model_rf.fit(X_train, y_train)
y_pred_rf  = model_rf.predict(X_test)
y_proba_rf = model_rf.predict_proba(X_test)[:, 1]
cv_rf = cross_val_score(model_rf, X_train, y_train, cv=5, scoring='roc_auc')
metrics_rf = {
    'accuracy': float(accuracy_score(y_test, y_pred_rf)),
    'precision': float(precision_score(y_test, y_pred_rf)),
    'recall': float(recall_score(y_test, y_pred_rf)),
    'f1': float(f1_score(y_test, y_pred_rf)),
    'auc': float(roc_auc_score(y_test, y_proba_rf)),
    'cv_auc_mean': float(cv_rf.mean()),
    'cv_auc_std': float(cv_rf.std()),
    'confusion_matrix': confusion_matrix(y_test, y_pred_rf).tolist(),
    'feature_importance': dict(zip(feature_names, model_rf.feature_importances_.tolist()))
}

print("\n--- COMPARAISON ---")
print(f"{'Métrique':<12} {'Logistic Reg':<15} {'Random Forest'}")
for k in ['accuracy','precision','recall','f1','auc']:
    winner = "<- LR" if metrics_lr[k] >= metrics_rf[k] else "<- RF"
    print(f"{k:<12} {metrics_lr[k]:<15.4f} {metrics_rf[k]:.4f}  {winner}")

os.makedirs('/home/claude/models', exist_ok=True)
joblib.dump(scaler,   '/home/claude/models/scaler.pkl')
joblib.dump(model_lr, '/home/claude/models/logistic_regression.pkl')
joblib.dump(model_rf, '/home/claude/models/random_forest.pkl', compress=3)

metadata = {
    'feature_names': feature_names,
    'n_features': len(feature_names),
    'n_train': int(len(X_train)),
    'n_test': int(len(X_test)),
    'models': {'logistic_regression': metrics_lr, 'random_forest': metrics_rf}
}
with open('/home/claude/models/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("\n✅ Tous les modèles sauvegardés dans /home/claude/models/")
