import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor


class Trainer:

    @staticmethod
    def train_failure_model(preprocessor, train, val=None):
        """
        Train a machine failure classifier
        """
        # Features and target
        X_train = train.drop(["Machine failure", "RUL"], axis=1)
        y_train = train["Machine failure"]

        if val is not None:
            X_val = val.drop(["Machine failure", "RUL"], axis=1)
            y_val = val["Machine failure"]

        # Pipeline with preprocessor and classifier
        model = Pipeline([
            ("prep", preprocessor),
            ("clf", XGBClassifier(
                n_estimators=300,
                max_depth=7,
                learning_rate=0.05,
                scale_pos_weight=5,
                use_label_encoder=False,
                eval_metric='logloss'  # avoids warning
            ))
        ])

        model.fit(X_train, y_train)
        return model



    @staticmethod
    def train_rul_model(preprocessor, train, val=None):
        """
        Train a Remaining Useful Life (RUL) regressor
        """
        # Features and target
        X_train = train.drop(["Machine failure", "RUL"], axis=1)
        y_train = train["RUL"]

        if val is not None:
            X_val = val.drop(["Machine failure", "RUL"], axis=1)
            y_val = val["RUL"]

        # Pipeline with preprocessor and regressor
        model = Pipeline([
            ("prep", preprocessor),
            ("reg", XGBRegressor(
                n_estimators=400,
                max_depth=8,
                learning_rate=0.03
            ))
        ])

        model.fit(X_train, y_train)
        return model
