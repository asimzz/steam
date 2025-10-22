import os
import json
import math


class KeySpace:
    def __init__(
        self,
        vocab_size,
    ):
        self.vocab_size = vocab_size

    def compute_key_space_size(self):
        pass

    def get_bits_of_security(self, log2_total_subsets):
        coefficient = 2 ** (log2_total_subsets - int(log2_total_subsets))
        exponent = int(log2_total_subsets)
        return coefficient, exponent


class XSIRKeySpace(KeySpace):
    def __init__(self, vocab_size, cluster_file):
        super().__init__(vocab_size)
        if os.path.exists(cluster_file):
            with open(cluster_file, "r") as f:
                self.clusters = json.load(f)
        else:
            raise ValueError(f"Cluster file {cluster_file} not found")

        self.cluster_size = len(self.clusters)

    def compute_key_space_size(self):
        return self.get_bits_of_security(self.cluster_size)


class KGWKeySpace(KeySpace):
    def __init__(self, vocab_size, gamma):
        super().__init__(vocab_size)
        self.gamma = gamma

    def compute_key_space_size(self):
        greenlist_size = int(self.gamma * self.vocab_size)
        n = self.vocab_size
        k = greenlist_size
        log2_total_subsets = (
            sum(math.log2(i) for i in range(1, n + 1))  # log2(n!)
            - sum(math.log2(i) for i in range(1, k + 1))  # log2(k!)
            - sum(math.log2(i) for i in range(1, n - k + 1))  # log2((n-k)!)
        )
        return self.get_bits_of_security(log2_total_subsets)

class SIRKeySpace(KeySpace):
    def __init__(self, vocab_size):
        super().__init__(vocab_size)
        
    def compute_key_space_size(self):
        return self.get_bits_of_security(self.vocab_size)