import pandas as pd
from preprocessing import Preprocessing
from trainer import Trainer
from evaluator import Evaluator

class PdMPipeline:

    @staticmethod
    def run_pipeline(data_path: str):
        """
        Run full PdM pipeline: load data, create RUL, split, preprocess, train, evaluate
        Returns trained failure and RUL models
        """

        df = Preprocessing.data_loading(data_path)

        df = Preprocessing.create_rul(df)


        train, val, test = Preprocessing.split_data(df)

        preprocessor, num_cols, cat_cols = Preprocessing.build_preprocessor(df)


        failure_model = Trainer.train_failure_model(preprocessor, train, val)
        rul_model = Trainer.train_rul_model(preprocessor, train, val)

        print("=== FAILURE MODEL EVALUATION ===")
        cls_report, roc_auc = Evaluator.evaluate_failure(failure_model, test)
        print(cls_report)
        print("ROC-AUC:", roc_auc)

        print("\n=== RUL MODEL EVALUATION ===")
        mae, rmse, r2 = Evaluator.evaluate_rul(rul_model, test)
        print("MAE:", mae)
        print("RMSE:", rmse)
        print("R2:", r2)

        return failure_model, rul_model
