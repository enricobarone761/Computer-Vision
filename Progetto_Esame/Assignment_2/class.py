from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC

#from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import StratifiedKFold


skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
X = np.array([istogramma for _, istogramma in lista_istogrammi])
y = np.array([classe for classe, _ in lista_istogrammi])

for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(f"Fold {fold + 1}")
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Logistic Regression
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    print("Logistic Regression trained.")

    # Random Forest
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    print("Random Forest trained.")

    # SVM
    svm = SVC(kernel='rbf')
    svm.fit(X_train, y_train)
    print("SVM trained.")

    # Linear SVM
    linear_svm = LinearSVC()
    linear_svm.fit(X_train, y_train)
    print("Linear SVM trained.")