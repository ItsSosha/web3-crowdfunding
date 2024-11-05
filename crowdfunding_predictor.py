import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


class CrowdfundingPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    def prepare_features(self, df):
        """
        Підготовка ознак для моделі
        """
        features = pd.DataFrame()

        # Основні числові характеристики
        features['funding_goal'] = df['funding_goal']
        features['campaign_duration'] = df['duration_days']
        features['reward_levels'] = df['reward_levels_count']

        # Текстові характеристики (приклад обробки)
        features['description_length'] = df['description'].str.len()
        features['has_video'] = df['has_video'].astype(int)

        # Часові характеристики
        features['is_weekend'] = df['launch_date'].dt.weekday >= 5
        features['launch_month'] = df['launch_date'].dt.month

        # Категорійні ознаки (one-hot encoding)
        category_dummies = pd.get_dummies(df['category'], prefix='category')
        features = pd.concat([features, category_dummies], axis=1)

        return features

    def train(self, training_data):
        """
        Навчання моделі на історичних даних
        """
        # Підготовка даних
        X = self.prepare_features(training_data)
        y = (training_data['funded_amount'] >= training_data['funding_goal']).astype(int)

        # Розділення на тренувальну і тестову вибірки
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Нормалізація даних
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Навчання моделі
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)

        # Оцінка якості
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.2f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

    def predict_success_probability(self, project_data):
        """
        Передбачення ймовірності успіху нового проєкту
        """
        if self.model is None:
            raise Exception("Model needs to be trained first!")

        # Підготовка даних нового проєкту
        X_new = self.prepare_features(pd.DataFrame([project_data]))
        X_new_scaled = self.scaler.transform(X_new)

        # Отримання ймовірності успіху
        success_probability = self.model.predict_proba(X_new_scaled)[0][1]

        return success_probability

    def save_model(self, filepath):
        """
        Збереження моделі
        """
        if self.model is None:
            raise Exception("No model to save!")
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler
        }, filepath)

    def load_model(self, filepath):
        """
        Завантаження збереженої моделі
        """
        saved_model = joblib.load(filepath)
        self.model = saved_model['model']
        self.scaler = saved_model['scaler']


# Приклад використання:
if __name__ == "__main__":
    # Створення тестових даних
    sample_data = {
        'funding_goal': [5000, 10000, 15000],
        'duration_days': [30, 45, 60],
        'reward_levels_count': [5, 7, 3],
        'description': ['Project A description', 'Project B description', 'Project C description'],
        'has_video': [True, False, True],
        'launch_date': pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01']),
        'category': ['Technology', 'Art', 'Games'],
        'funded_amount': [6000, 8000, 20000]
    }

    df = pd.DataFrame(sample_data)

    # Ініціалізація та навчання предиктора
    predictor = CrowdfundingPredictor()
    predictor.train(df)

    # Приклад передбачення для нового проєкту
    new_project = {
        'funding_goal': 7500,
        'duration_days': 40,
        'reward_levels_count': 6,
        'description': 'New project description',
        'has_video': True,
        'launch_date': pd.to_datetime('2024-04-01'),
        'category': 'Technology'
    }

    success_prob = predictor.predict_success_probability(new_project)
    print(f"\nPredicted success probability: {success_prob:.2f}")