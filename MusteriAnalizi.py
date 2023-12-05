#%%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import numpy as np
#%%
df_train= pd.read_csv("C:/Users/Msi/Desktop/CALISMALAR/Musteri/train.csv")
#%%
print(df_train.shape)
df_train.info()
#%%
df_train.nunique()
#%%
df_train.nunique()
df_train['y'] = df_train['y'].map({'no': 0, 'yes': 1})
#%%
labels = 'Reach', 'No Reach'
sizes = [df_train.y[df_train['y']==1].count(), df_train.y[df_train['y']==0].count()]
explode = (0, 0.1)
fig1, ax1 = plt.subplots(figsize=(10, 8))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')
plt.title("Reach vs No Reach", size = 20)
plt.show()
#%%
fig, axarr = plt.subplots(2, 2, figsize=(20, 12))
#sns.countplot(x='job', hue = 'y',data = df_train, ax=axarr[0][0])
sns.countplot(x='loan', hue = 'y',data = df_train, ax=axarr[0][1])
sns.countplot(x='education', hue = 'y',data = df_train, ax=axarr[1][0])
sns.countplot(x='contact', hue = 'y',data = df_train, ax=axarr[1][1])

#%%
df_train['job'] = df_train['job'].astype('category').cat.codes
df_train['marital'] = df_train['marital'].astype('category').cat.codes
df_train['education'] = df_train['education'].astype('category').cat.codes
df_train['default'] = df_train['default'].astype('category').cat.codes
df_train['housing'] = df_train['housing'].astype('category').cat.codes
df_train['loan'] = df_train['loan'].astype('category').cat.codes
df_train['contact'] = df_train['contact'].astype('category').cat.codes
df_train['month'] = df_train['month'].astype('category').cat.codes
df_train['poutcome'] = df_train['poutcome'].astype('category').cat.codes


#%%
X = df_train.drop('y', axis=1)
y=df_train['y'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
#%%
unique_values, counts = np.unique(y_train, return_counts=True)
class_distribution = dict(zip(unique_values, counts))
print(class_distribution)
imbalance_ratio = class_distribution[0] / class_distribution[1]
print("Imbalance Ratio:", imbalance_ratio)
threshold = 2
if imbalance_ratio > threshold:
    # Initialize SMOTE
    smote = SMOTE(sampling_strategy='auto', random_state=42)

    # Fit and apply SMOTE to the training data
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Check the class distribution after SMOTE
    unique_values, counts = np.unique(y_train_resampled, return_counts=True)
    resampled_class_distribution = dict(zip(unique_values, counts))

    print("\nClass Distribution After SMOTE:")
    print(resampled_class_distribution)

    # Update the training data with the resampled data
    X_train = X_train_resampled
    y_train = y_train_resampled
    print("\nSMOTE Applied. Training data resampled.")
else:
    print("\nNo significant class imbalance. SMOTE not applied.")
#
#%%
#Logistic
logreg_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))  # max_iter değerini artırabilirsiniz
logreg_model.fit(X_train, y_train)
logreg_predictions = logreg_model.predict(X_test)
logreg_accuracy = accuracy_score(y_test, logreg_predictions)
print("Logistic Regression Accuracy:", logreg_accuracy)
print("Logistic Regression Classification Report:")
print(classification_report(y_test, logreg_predictions))
#%%
#XGB
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_predictions)
print("\nXGBoost Accuracy:", xgb_accuracy)
print("XGBoost Classification Report:")
print(classification_report(y_test, xgb_predictions))
#%%
#Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print("\nRandom Forest Accuracy:", rf_accuracy)
print("Random Forest Classification Report:")
print(classification_report(y_test, rf_predictions))
#%%
#Desicion Tree
de_model = DecisionTreeClassifier(max_depth=53)
de_model.fit(X_train, y_train)
de_model_predictions = rf_model.predict(X_test)
de_model_accuracy = accuracy_score(y_test, rf_predictions)
print("\nDecision Tree Accuracy:", de_model_accuracy)
print("Decision Tree Classification Report:")
print(classification_report(y_test, de_model_predictions))
#%%
# Logistic Regression ROC Curve
logreg_probs = logreg_model.predict_proba(X_test)[:, 1]
logreg_fpr, logreg_tpr, _ = roc_curve(y_test, logreg_probs)
logreg_auc = roc_auc_score(y_test, logreg_probs)
# XGBoost ROC Curve
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_probs)
xgb_auc = roc_auc_score(y_test, xgb_probs)
# Random Forest ROC Curve
rf_probs = rf_model.predict_proba(X_test)[:, 1]
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
rf_auc = roc_auc_score(y_test, rf_probs)

# Decison  Tree  ROC Curve
de_probs = de_model.predict_proba(X_test)[:, 1]
de_fpr, de_tpr, _ = roc_curve(y_test, de_probs)
de_auc = roc_auc_score(y_test, de_probs)
#%%
plt.figure(figsize=(10, 6))
plt.plot(logreg_fpr, logreg_tpr, label=f'Logistic Regression (AUC = {logreg_auc:.2f})')
plt.plot(xgb_fpr, xgb_tpr, label=f'XGBoost (AUC = {xgb_auc:.2f})')
plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_auc:.2f})')
plt.plot(de_fpr, de_tpr, label=f'Decision Tree (AUC = {de_auc:.2f})')

plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
#%%
auc_values = [logreg_auc, xgb_auc, rf_auc, de_auc]
model_names = ['Logistic Regression', 'XGBoost', 'Random Forest', 'Decision Tree']

# Find the index of the model with the highest AUC
best_model_index = auc_values.index(max(auc_values))

# Print the best model and its AUC
best_model_name = model_names[best_model_index]
best_model_auc = auc_values[best_model_index]

print(f"The best model is {best_model_name} with AUC = {best_model_auc:.2f}")

#%%
df_test= pd.read_csv("C:/Users/Msi/Desktop/CALISMALAR/Musteri/test.csv")
df_test['job'] = df_test['job'].astype('category').cat.codes
df_test['marital'] = df_test['marital'].astype('category').cat.codes
df_test['education'] = df_test['education'].astype('category').cat.codes
df_test['default'] = df_test['default'].astype('category').cat.codes
df_test['housing'] = df_test['housing'].astype('category').cat.codes
df_test['loan'] = df_test['loan'].astype('category').cat.codes
df_test['contact'] = df_test['contact'].astype('category').cat.codes
df_test['month'] = df_test['month'].astype('category').cat.codes
df_test['poutcome'] = df_test['poutcome'].astype('category').cat.codes

#%%
X_test = df_test.drop('y', axis=1)
#%%
y_test_pred = xgb_model.predict(X_test)
#%%
df_test['predicted_target'] = y_test_pred
#%%
print(df_test[['y', 'predicted_target']])
#%%
df_test['predicted_target'] = ['no' if pred == 0 else 'yes' for pred in y_test_pred]
conf_matrix = confusion_matrix(df_test['y'], df_test['predicted_target'])
#%%
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Tahmin Edilen Etiketler')
plt.ylabel('Gerçek Etiketler')
plt.title('Confusion Matrix')
plt.show()
#%%
conf_matrix = confusion_matrix(df_test['y'], df_test['predicted_target'])
accuracy = accuracy_score(df_test['y'], df_test['predicted_target'])
classification_rep = classification_report(df_test['y'], df_test['predicted_target'])

print('Confusion Matrix:')
print(conf_matrix)
print('Accuracy:', accuracy)
print('Classification Report:')
print(classification_rep)

#%%
feature_importance = xgb_model.feature_importances_
feature_names = X_train.columns
feature_importance = list(zip(feature_names, feature_importance))
feature_importance.sort(key=lambda x: x[1], reverse=True)
#%%
plt.figure(figsize=(10, 6))
sns.barplot(x=[x[1] for x in feature_importance], y=[x[0] for x in feature_importance], palette="viridis")
plt.title('XGBoost Feature Importance')
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.show()