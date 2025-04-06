# Data preprocessing  
from sklearn.impute import SimpleImputer  
imputer = SimpleImputer(strategy='median')  
X[['glucose', 'bmi']] = imputer.fit_transform(X[['glucose', 'bmi']])  

# Model training  
from sklearn.ensemble import RandomForestClassifier  
model = RandomForestClassifier(max_depth=5, n_estimators=100, random_state=42)  
model.fit(X_train, y_train)  

# Evaluation  
from sklearn.metrics import classification_report  
print(classification_report(y_test, predictions))  