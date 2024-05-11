from typing import Dict
import json 

ROUND = int

class BlackBox:
    def __init__(self):
        self.current_fl_round: ROUND = 0
        self.client_count_record: Dict[ROUND, int] = {}
        self.model_integer_range_record: Dict[ROUND, int] = {}
        self.modulus_record: Dict[ROUND, int] = {}
        self.report = {}

    def record_setup(
        self,
        round: ROUND,
        online_clients: int,
        range: int,
    ) -> None:
        """Collect internal states for setup phase of SecAgg on given federated round."""
        self.current_fl_round = round
        self.client_count_record[round] = online_clients
        self.model_integer_range_record[round] = range

    def record_modulus(self, round: ROUND, modulus: int) -> None:
        """Record at the moment of its use near the end of SecAgg."""
        self.modulus[round] = modulus

    def __repr__(self):
        return "A blackbox for federated learning with secure aggregation. Records key internal states."

    def save(self, path):
        with open('report.json', 'w') as json_file:
            json.dump(self.report, json_file)

