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

        # Текстові характеристики (приклад обробки)
        features['description_length'] = df['description'].str.len()

        # Часові характеристики
        features['is_weekend'] = df['launch_date'].dt.weekday >= 5
        features['launch_month'] = df['launch_date'].dt.month

        # Категорійні ознаки (one-hot encoding)
        # TODO: Fix later
        # category_dummies = pd.get_dummies(df['category'], prefix='category')
        # features = pd.concat([features, category_dummies], axis=1)

        return features

    def load_and_prepare_data(self, csv_path):
        """
        Завантажує та готує дані з CSV файлу з користувацькими назвами колонок
        """
        # Читаємо CSV
        df = pd.read_csv(csv_path)

        # Словник для перейменування колонок
        # Формат: 'стара_назва': 'нова_назва'
        # column_mapping = {
        #     'Project_Name': 'project_name',
        #     'Goal_Amount_USD': 'funding_goal',
        #     'Campaign_Duration': 'duration_days',
        #     'Number_of_Rewards': 'reward_levels',
        #     'Video_Present': 'has_video',
        #     'Project_Category': 'category',
        #     'Project_Description': 'description',
        #     'Launch_Date': 'launch_date',
        #     'Amount_Raised': 'final_amount'
        # }

        column_mapping = {
            "goal_usd": 'funding_goal',
            'duration': 'duration_days',
            "name": 'description',
            "launched_at": 'launch_date',
            "main_category": 'category',
            "usd_pledged": 'funded_amount'
        }

        # Перейменовуємо тільки ті колонки, які є в датафреймі
        existing_columns = {old: new for old, new in column_mapping.items()
                            if old in df.columns}
        df = df.rename(columns=existing_columns)

        # Список необхідних колонок
        required_columns = [
            'funding_goal',
            'duration_days',
            'category',
            'description',
            'launch_date',
            'funded_amount'
        ]

        # Перевіряємо наявність необхідних колонок
        missing_columns = [col for col in required_columns
                           if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Видаляємо непотрібні колонки
        df = df[required_columns]

        # Конвертація дат
        df['launch_date'] = pd.to_datetime(df['launch_date'])

        # Конвертація числових значень
        df['funding_goal'] = pd.to_numeric(df['funding_goal'], errors='coerce')
        df['funded_amount'] = pd.to_numeric(df['funded_amount'], errors='coerce')
        df['duration_days'] = pd.to_numeric(df['duration_days'], errors='coerce')

        # Очищення категорій
        df['category'] = df['category'].str.strip()

        # Розрахунок успішності
        df['is_successful'] = (df['funded_amount'] >= df['funding_goal']).astype(int)

        # Видалення рядків з відсутніми значеннями
        df = df.dropna(subset=['funding_goal', 'funded_amount'])

        # Додаткові характеристики
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
        X = self.prepare_features(training_data)
        y = (training_data['funded_amount'] >= training_data['funding_goal']).astype(int)

        print("Training data shape: ", X.head())

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

        X_new = self.prepare_features(pd.DataFrame([project_data]))

        print("Predict data shape: ", X_new.head())
        X_new_scaled = self.scaler.transform(X_new)

        success_probability = self.model.predict_proba(X_new_scaled)[0][1]

        return success_probability



    # def predict_success_probability(self, project_data):
    #     # Створюємо DataFrame з одним проєктом
    #     df = pd.DataFrame([project_data])
    #
    #     # Готуємо базові характеристики
    #     X_new = self.prepare_features(df)
    #
    #     # Отримуємо всі категорії, які були при навчанні
    #     training_categories = [col for col in self.model.feature_names_in_ if col.startswith('category_')]
    #
    #     # Додаємо всі відсутні категорії та заповнюємо їх нулями
    #     for category in training_categories:
    #         if category not in X_new.columns:
    #             X_new[category] = 0
    #
    #     # Встановлюємо 1 для категорії проєкту
    #     project_category = f"category_{project_data['category'].lower()}"
    #     if project_category in training_categories:
    #         X_new[project_category] = 1
    #
    #     # Упорядковуємо колонки як при навчанні
    #     X_new = X_new[self.model.feature_names_in_]
    #
    #     # Нормалізація та предикція
    #     X_new_scaled = self.scaler.transform(X_new)
    #     return self.model.predict_proba(X_new_scaled)[0][1]

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
    pd.set_option('display.max_columns', 20)
    predictor = CrowdfundingPredictor()
    df = predictor.load_and_prepare_data("Kickstarter_projects_Feb19.csv")
    predictor.train(df)

    # unsuccessfulCrowdfunding = predictor.prepare_features(df[df["is_successful"] == False].iloc[0])
    # successfulCrowdfunding = predictor.prepare_features(df[df["is_successful"] == True].iloc[0])
    unsuccessfulCrowdfunding = df[df["is_successful"] == False].iloc[5]
    successfulCrowdfunding = df[df["is_successful"] == True].iloc[10]
    print(unsuccessfulCrowdfunding)
    print(successfulCrowdfunding)

    print(predictor.predict_success_probability(unsuccessfulCrowdfunding))
    print(predictor.predict_success_probability(successfulCrowdfunding))

    # Створення тестових даних
    # sample_data = {
    #     'funding_goal': [5000, 10000, 15000],
    #     'duration_days': [30, 45, 60],
    #     'reward_levels_count': [5, 7, 3],
    #     'description': ['Project A description', 'Project B description', 'Project C description'],
    #     'has_video': [True, False, True],
    #     'launch_date': pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01']),
    #     'category': ['Technology', 'Art', 'Games'],
    #     'funded_amount': [6000, 8000, 20000]
    # }
    #
    # df = pd.DataFrame(sample_data)
    #
    # # Ініціалізація та навчання предиктора
    # predictor = CrowdfundingPredictor()
    # predictor.train(df)
    #
    # # Приклад передбачення для нового проєкту
    # new_project = {
    #     'funding_goal': 7500,
    #     'duration_days': 40,
    #     'reward_levels_count': 6,
    #     'description': 'New project description',
    #     'has_video': True,
    #     'launch_date': pd.to_datetime('2024-04-01'),
    #     'category': 'Technology'
    # }
    #
    # success_prob = predictor.predict_success_probability(new_project)
    # print(f"\nPredicted success probability: {success_prob:.2f}")
