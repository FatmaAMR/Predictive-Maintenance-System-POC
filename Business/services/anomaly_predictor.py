import joblib

class AnomalyPredictor:
    def __init__(self):
        
        
        # trained models
        self.classifier = joblib.load("business/ml/models/failure_classifier.pkl")
        
        self.rul_predictor = joblib.load("business/ml/models/rul_regressor.pkl")
        
        self.preprocessor = joblib.load("business/ml/models/preprocessor.pkl")
        
        
        self.max_rul = 300
        

    def predict_failure(self, features):
        """
        Predict machine failure (0/1)
        features: pandas DataFrame or 2D array
        """
        features = self.preprocessor.transform(features)
        return self.classifier.predict(features)



    def predict_failure_proba(self, features):
        """
        Predict failure probability (useful for risk assessment)
        """
        features = self.preprocessor.transform(features)
        return self.classifier.predict_proba(features)[:, 1]

    def predict_RUL(self, features):
        """
        Predict Remaining Useful Life
        """
        
        return self.rul_predictor.predict(features)
    
    
    def generate_insight(self, df):
        """
        Input:
            df: pandas DataFrame with raw sensor inputs
        
        Output:
            Dictionary containing failure prob, RUL, category, and recommendation
        """

        # 1. Make predictions
        failure_pred = int(self.predict_failure(df)[0])
        failure_prob = float(self.predict_failure_proba(df)[0])
        rul_value = float(self.predict_RUL(df)[0])

        # 2. Categorize RUL
        risk_level = self.categorize_rul(rul_value)

        # 3. Business recommendations
        action_map = {
            "Critical": "Stop machine immediately",
            "High": "Schedule maintenance ASAP",
            "Medium": "Monitor closely",
            "Low": "Normal operation"
        }

        # 4. Final output
        insight = {
            "failure_prediction": failure_pred,
            "failure_probability": failure_prob,
            "predicted_RUL": rul_value,
            "risk_level": risk_level,
            "recommended_action": action_map[risk_level]
        }

        return insight