from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# Veri yükleme
data = load_iris()
X = data.data
y = data.target

# Train / test bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = GaussianNB()
model.fit(X_train, y_train)

# Tahmin
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
