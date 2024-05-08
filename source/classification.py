from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

def logistic_regression(X_train, X_test, y_train, y_test):
    # Train a Logistic Regression model
    classifier = LogisticRegression(random_state=42)
    classifier.fit(X_train, y_train)

    # Evaluate the model on test data
    y_pred = classifier.predict(X_test)

    return y_pred, y_test

def random_forest(X_train, X_test, y_train, y_test):
    # Train a Random Forest model
    classifier = RandomForestClassifier(random_state=42)
    classifier.fit(X_train, y_train)

    # Evaluate the model on test data
    y_pred = classifier.predict(X_test)

    return y_pred, y_test

def support_vector_machine(X_train, X_test, y_train, y_test):
    # Train a Support Vector Machine model
    classifier = SVC(random_state=42, probability=True)
    classifier.fit(X_train, y_train)

    # Evaluate the model on test data
    y_pred = classifier.predict(X_test)

    return y_pred, y_test

def gradient_boosting(X_train, X_test, y_train, y_test):
    # Train a Gradient Boosting model
    classifier = XGBClassifier(random_state=42)
    classifier.fit(X_train, y_train)

    # Evaluate the model on test data
    y_pred = classifier.predict(X_test)

    return y_pred, y_test

def neural_network(X_train, X_test, y_train, y_test):
    # Train a Neural Network model
    classifier = MLPClassifier(random_state=42)
    classifier.fit(X_train, y_train)

    # Evaluate the model on test data
    y_pred = classifier.predict(X_test)

    return y_pred, y_test

def k_nearest_neighbors(X_train, X_test, y_train, y_test):
    # Train a K-Nearest Neighbors model
    classifier = KNeighborsClassifier()
    classifier.fit(X_train, y_train)

    # Evaluate the model on test data
    y_pred = classifier.predict(X_test)

    return y_pred, y_test

def naive_bayes(X_train, X_test, y_train, y_test):
    # Train a Naive Bayes model
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    # Evaluate the model on test data
    y_pred = classifier.predict(X_test)

    return y_pred, y_test

def decision_tree(X_train, X_test, y_train, y_test):
    # Train a Decision Tree model
    classifier = DecisionTreeClassifier(random_state=42)
    classifier.fit(X_train, y_train)

    # Evaluate the model on test data
    y_pred = classifier.predict(X_test)

    return y_pred, y_test

def ada_boost(X_train, X_test, y_train, y_test):
    # Train an Ensemble method (AdaBoost) model
    classifier = AdaBoostClassifier(random_state=42)
    classifier.fit(X_train, y_train)

    # Evaluate the model on test data
    y_pred = classifier.predict(X_test)

    return y_pred, y_test