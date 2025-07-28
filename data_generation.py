import numpy as np
import random
import pandas as pd

class DataGenerator:
    def __init__(self, num_content=10):
        self.num_content = num_content
        self.network_conditions = []
        self.user_engagement = []

    def simulate_network_conditions(self, num_samples=100):
        """
        Simulates network conditions including latency, bandwidth, and packet loss.
        """
        latencies = np.random.normal(loc=50, scale=10, size=num_samples)  # Latency in ms
        bandwidths = np.random.normal(loc=100, scale=20, size=num_samples)  # Bandwidth in Mbps
        packet_losses = np.random.uniform(0, 0.05, size=num_samples)  # Packet loss in percentage
        
        # Ensure non-negative values
        latencies = np.abs(latencies)
        bandwidths = np.abs(bandwidths)

        self.network_conditions = pd.DataFrame({
            'Latency (ms)': latencies,
            'Bandwidth (Mbps)': bandwidths,
            'Packet Loss (%)': packet_losses
        })

    def simulate_user_engagement(self):
        """
        Simulates user engagement patterns including content requests and engagement scores.
        """
        content_ids = [f'Content_{i+1}' for i in range(self.num_content)]
        
        requests = [random.randint(50, 500) for _ in range(self.num_content)]  # Random requests per content
        engagement_scores = [random.randint(0, 100) for _ in range(self.num_content)]  # Engagement scores
        
        self.user_engagement = pd.DataFrame({
            'Content ID': content_ids,
            'Requests': requests,
            'Engagement Score': engagement_scores
        })

    def display_data(self):
        """
        Displays the generated network conditions and user engagement data.
        """
        print("Network Conditions:")
        print(self.network_conditions.describe())
        
        print("\nUser Engagement:")
        print(self.user_engagement)

if __name__ == "__main__":
    # Create an instance of DataGenerator
    data_gen = DataGenerator(num_content=10)

    # Simulate network conditions
    data_gen.simulate_network_conditions(num_samples=100)

    # Simulate user engagement
    data_gen.simulate_user_engagement()

    # Display the generated data
    data_gen.display_data()
