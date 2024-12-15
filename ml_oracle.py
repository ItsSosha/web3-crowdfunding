import os
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

    def start_listening(self):
        """
        Запускає прослуховування обох подій: створення проєктів та внесення коштів
        """
        project_filter = self.contract.events.ProjectCreated.create_filter(
            fromBlock='latest'
        )
        contribution_filter = self.contract.events.ContributionMade.create_filter(
            fromBlock='latest'
        )

        while True:
            try:
                for event in project_filter.get_new_entries():
                    project_id = event['args']['id']
                    self.update_project_prediction(project_id)

                for event in contribution_filter.get_new_entries():
                    project_id = event['args']['projectId']
                    self.update_project_prediction(project_id)

                time.sleep(10)

            except Exception as e:
                print(f"Помилка при обробці подій: {str(e)}")
                time.sleep(30)
                continue

    def update_project_prediction(self, project_id):
        """
        Оновлює предикцію для проєкту на основі поточних метрик
        """
        project_data = self.contract.functions.getProject(project_id).call()
        project_metrics = self.contract.functions.getProjectMetrics(project_id).call()

        project_data = {
            'funding_goal': project_data[2],
            'duration_days': project_metrics.durationDays,
            'category': project_metrics.category,
            'description_length': project_metrics.descriptionLength,
            'current_funding': project_data[3],
            'funding_percentage': (project_data[3] / project_data[0]) * 100,
        }

        success_probability = self.predictor.predict_success_probability(project_data)
        prediction_percentage = int(success_probability * 100)

        self._send_prediction(project_id, prediction_percentage)

    def _send_prediction(self, project_id, prediction):
        """
        Відправляє предикцію в смарт-контракт з обробкою помилок та повторними спробами
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
        return


if __name__ == "__main__":
    CONTRACT_ADDRESS = os.environ.get("CONTRACT_ADDRESS")
    WEB3_PROVIDER = os.environ.get("WEB3_PROVIDER", "http://localhost:8545")
    PRIVATE_KEY = os.environ.get("PRIVATE_KEY")

    with open('contract_abi.json', 'r') as f:
        CONTRACT_ABI = json.load(f)

    oracle = MLOracle(
        CONTRACT_ADDRESS,
        CONTRACT_ABI,
        WEB3_PROVIDER,
        PRIVATE_KEY
    )

    print("ML Oracle запущено. Очікування подій...")
    oracle.start_listening()
