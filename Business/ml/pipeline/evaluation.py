import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class Evaluator:

    @staticmethod
    def evaluate_failure(model, test):
        """
        Evaluate classifier performance on test set
        Returns classification report and ROC-AUC score
        """
        X_test = test.drop(["Machine failure", "RUL"], axis=1)
        y_test = test["Machine failure"]

        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

        cls_report = classification_report(y_test, preds)
        roc_auc = roc_auc_score(y_test, proba)

        return cls_report, roc_auc

    @staticmethod
    def evaluate_rul(model, test):
        """
        Evaluate RUL regression model on test set
        Returns MAE, RMSE, and R2 score
        """
        X_test = test.drop(["Machine failure", "RUL"], axis=1)
        y_test = test["RUL"]

        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        return mae, rmse, r2
