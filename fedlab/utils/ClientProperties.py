import pickle
import os
import random


def load_client_profile(file_path):
    """For Simulation Mode: load client profiles/traces

    Args:
        file_path (string): File path for the client profiles/traces

    Returns:
        dictionary: Return the client profiles/traces

    """
    global_client_profile = {}
    if os.path.exists(file_path):
        with open(file_path, "rb") as fin:
            # {client_id: [compute, bandwidth]}
            # compute: Gflops
            # bandwidth: Mbps
            global_client_profile = pickle.load(fin)
    return global_client_profile


class ClientPropertyManager:
    def __init__(self, client_profile_path: str, total_client_num: int):
        all_client_profiles = load_client_profile(file_path=client_profile_path)
        self.avg_comm = sum([val['communication'] for val in all_client_profiles.values()]) / len(all_client_profiles)
        self.avg_comp = sum([val['computation'] for val in all_client_profiles.values()]) / len(all_client_profiles)
        selected_keys = random.sample(list(all_client_profiles.keys()), total_client_num)
        self.client_profiles = [all_client_profiles[key] for key in selected_keys]

    def get_client_profile(self, client_id):
        """
        Args:
            client_id:
        Returns:
            {"computation": xxx, "communication": xxx}
            Gflops, Mb/s
        """
        client_id = client_id % len(self.client_profiles)
        return  self.client_profiles[client_id]

    def get_random_profile(self, num):
        return random.sample(self.client_profiles, num)