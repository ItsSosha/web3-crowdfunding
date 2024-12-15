import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


class CrowdfundingPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.categories = None

    def prepare_features(self, df, training=False):
        """
        Підготовка ознак для моделі
        """
        features = pd.DataFrame()

        features['funding_goal'] = df['funding_goal']
        features['campaign_duration'] = df['duration_days']
        features['description_length'] = df['description'].str.len()

        features['is_weekend'] = df['launch_date'].dt.weekday >= 5
        features['launch_month'] = df['launch_date'].dt.month

        if training:
            self.categories = sorted(df['category'].unique())
            category_dummies = pd.get_dummies(df['category'], prefix='category')
        else:
            category_dummies = pd.DataFrame(0, index=df.index,
                                            columns=[f'category_{cat.lower()}' for cat in self.categories])
            for cat in df['category'].unique():
                if cat in self.categories:
                    category_dummies[f'category_{cat.lower()}'] = (df['category'] == cat).astype(int)

        features = pd.concat([features, category_dummies], axis=1)
        return features

    def load_and_prepare_data(self, csv_path):
        """
        Завантажує та готує дані з CSV файлу з користувацькими назвами колонок
        """
        df = pd.read_csv(csv_path)

        column_mapping = {
            "goal_usd": 'funding_goal',
            'duration': 'duration_days',
            "name": 'description',
            "launched_at": 'launch_date',
            "main_category": 'category',
            "usd_pledged": 'funded_amount'
        }

        existing_columns = {old: new for old, new in column_mapping.items()
                            if old in df.columns}
        df = df.rename(columns=existing_columns)

        required_columns = [
            'funding_goal',
            'duration_days',
            'category',
            'description',
            'launch_date',
            'funded_amount'
        ]

        missing_columns = [col for col in required_columns
                           if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        df = df[required_columns]

        df['launch_date'] = pd.to_datetime(df['launch_date'])

        df['funding_goal'] = pd.to_numeric(df['funding_goal'], errors='coerce')
        df['funded_amount'] = pd.to_numeric(df['funded_amount'], errors='coerce')
        df['duration_days'] = pd.to_numeric(df['duration_days'], errors='coerce')

        df['category'] = df['category'].str.strip()

        df['is_successful'] = (df['funded_amount'] >= df['funding_goal']).astype(int)

        df = df.dropna(subset=['funding_goal', 'funded_amount'])

        df['description_length'] = df['description'].str.len()
        df['launch_month'] = df['launch_date'].dt.month
        df['launch_day_of_week'] = df['launch_date'].dt.dayofweek

        print(f"Successfully processed {len(df)} rows of data")

        print(df.head())

        return df

    def train(self, training_data):
        """
        Навчання моделі на історичних даних
        """
        # Підготовка даних
        X = self.prepare_features(training_data, True)
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

    def save_model(self, filepath):
        """
        Збереження моделі
        """
        if self.model is None:
            raise Exception("No model to save!")
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'categories': self.categories
        }, filepath)

    def load_model(self, filepath):
        """
        Завантаження збереженої моделі
        """
        saved_model = joblib.load(filepath)
        self.model = saved_model['model']
        self.scaler = saved_model['scaler']
        self.categories = saved_model['categories']


if __name__ == "__main__":
    predictor = CrowdfundingPredictor()
    trainDf = predictor.load_and_prepare_data("Kickstarter_projects_Feb19.csv")
    predictor.train(trainDf)
    predictor.save_model("model.joblib")