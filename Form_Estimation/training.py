import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# some basic classification!

# TODO: Maybe just pass in file path? small fix for later
def train_classifier_km():

    data_from_csv = pd.read_csv("training.csv")

    print(data_from_csv.head(5))

    rows = data_from_csv.iloc[:, :-1].values  # all rows except last
    exercise_col = data_from_csv["exercise"]

    X_train, X_test, y_train, y_test = train_test_split(
        rows, exercise_col, test_size=0.20, random_state=25
    )

    print(X_train)
    print(y_train)

    kn_model = KNeighborsClassifier(n_neighbors=5)  # instantiate knn
    kn_model.fit(X_train, y_train)  # fit training data
    kn_predict = kn_model.predict(X_test)  # prediction on separated test data
    print(accuracy_score(kn_predict, y_test))
    print(confusion_matrix(kn_predict, y_test))
    print(classification_report(kn_predict, y_test))

    return kn_predict
