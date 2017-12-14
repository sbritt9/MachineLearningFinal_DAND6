from sklearn.metrics import accuracy_score


def ScorePredictions(name, labels, predictor):
    print("=========================================================================")
    print("Estimator:",name)
    print("Accuracy Score:", accuracy_score(labels, predictor))
    print("=========================================================================")

