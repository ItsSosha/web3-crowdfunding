from web3 import Web3
from eth_account import Account
import json
import time
from crowdfunding_predictor import CrowdfundingPredictor


class MLOracle:
    def __init__(self, contract_address, contract_abi, web3_provider, private_key):
        self.web3 = Web3(Web3.HTTPProvider(web3_provider))
        self.contract = self.web3.eth.contract(
            address=contract_address,
            abi=contract_abi
        )
        self.account = Account.from_key(private_key)
        self.predictor = CrowdfundingPredictor()
        self.predictor.load_model('model.joblib')

    def listen_for_new_projects(self):
        """
        Слухає події створення нових проєктів та надає предикції
        """
        project_filter = self.contract.events.ProjectCreated.create_filter(
            fromBlock='latest'
        )

        while True:
            for event in project_filter.get_new_entries():
                project_id = event['args']['id']
                self.process_new_project(project_id)
            time.sleep(10)  # Перевірка кожні 10 секунд

    def process_new_project(self, project_id):
        """
        Обробка нового проєкту та надання предикції
        """
        # Отримання метрик проєкту
        metrics = self.contract.functions.getProjectMetrics(project_id).call()

        # Підготовка даних для ML-моделі
        project_data = {
            'funding_goal': metrics[0],
            'duration_days': metrics[1],
            'reward_levels_count': metrics[2],
            'has_video': metrics[3],
            'category': metrics[4],
            'description_length': metrics[5]
        }

        # Отримання предикції
        success_probability = self.predictor.predict_success_probability(project_data)
        prediction_percentage = int(success_probability * 100)

        # Відправка предикції в смарт-контракт
        self.send_prediction(project_id, prediction_percentage)

    def send_prediction(self, project_id, prediction):
        """
        Відправка предикції в смарт-контракт
        """
        nonce = self.web3.eth.get_transaction_count(self.account.address)

        transaction = self.contract.functions.setProjectPrediction(
            project_id,
            prediction
        ).build_transaction({
            'from': self.account.address,
            'gas': 2000000,
            'gasPrice': self.web3.eth.gas_price,
            'nonce': nonce,
        })

        signed_txn = self.web3.eth.account.sign_transaction(
            transaction,
            self.account.key
        )
        tx_hash = self.web3.eth.send_raw_transaction(signed_txn.rawTransaction)
        self.web3.eth.wait_for_transaction_receipt(tx_hash)


if __name__ == "__main__":
    # Конфігурація
    CONTRACT_ADDRESS = "YOUR_CONTRACT_ADDRESS"
    WEB3_PROVIDER = "http://localhost:8545"  # або інший провайдер
    PRIVATE_KEY = "YOUR_PRIVATE_KEY"

    with open('contract_abi.json', 'r') as f:
        CONTRACT_ABI = json.load(f)

    # Запуск оракула
    oracle = MLOracle(
        CONTRACT_ADDRESS,
        CONTRACT_ABI,
        WEB3_PROVIDER,
        PRIVATE_KEY
    )
    oracle.listen_for_new_projects()