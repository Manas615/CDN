import random
import time
from collections import defaultdict

class PerformanceMetrics:
    def __init__(self, num_servers):
        self.num_servers = num_servers
        self.server_usage = {i: {'cpu': 0, 'memory': 0} for i in range(num_servers)}
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_delivery_time = 0
        self.total_requests = 0
        self.total_engagement_score = 0
        self.total_cost = 0
        self.timeout_requests = 0
        self.link_costs = defaultdict(lambda: random.uniform(0.1, 1.0))

    def update_server_usage(self, server_id, cpu_usage, memory_usage):
        self.server_usage[server_id]['cpu'] = cpu_usage
        self.server_usage[server_id]['memory'] = memory_usage

    def record_cache_hit(self):
        self.cache_hits += 1
        self.total_requests += 1

    def record_cache_miss(self):
        self.cache_misses += 1
        self.total_requests += 1

    def record_delivery_time(self, time):
        self.total_delivery_time += time
        self.total_requests += 1

    def record_engagement_score(self, score):
        self.total_engagement_score += score

    def record_request_cost(self, path):
        cost = sum(self.link_costs[(path[i], path[i+1])] for i in range(len(path)-1))
        self.total_cost += cost

    def record_timeout(self):
        self.timeout_requests += 1
        self.total_requests += 1

    def get_cache_hit_rate(self):
        if self.total_requests == 0:
            return 0
        return self.cache_hits / self.total_requests

    def get_average_delivery_time(self):
        if self.total_requests == 0:
            return 0
        return self.total_delivery_time / self.total_requests

    def get_average_engagement_score(self):
        if self.total_requests == 0:
            return 0
        return self.total_engagement_score / self.total_requests

    def get_average_cost(self):
        if self.total_requests == 0:
            return 0
        return self.total_cost / self.total_requests

    def get_timeout_rate(self):
        if self.total_requests == 0:
            return 0
        return self.timeout_requests / self.total_requests

    def get_metrics_summary(self):
        return {
            "Server Usage": self.server_usage,
            "Cache Hit Rate": self.get_cache_hit_rate(),
            "Average Delivery Time": self.get_average_delivery_time(),
            "Average Engagement Score": self.get_average_engagement_score(),
            "Average Cost per Request": self.get_average_cost(),
            "Network Congestion (Timeout Rate)": self.get_timeout_rate()
        }

def simulate_cdn_requests(metrics, num_requests=1000):
    for _ in range(num_requests):
        # Simulate server usage
        for server_id in range(metrics.num_servers):
            cpu_usage = random.uniform(0, 100)
            memory_usage = random.uniform(0, 100)
            metrics.update_server_usage(server_id, cpu_usage, memory_usage)

        # Simulate cache hit/miss
        if random.random() < 0.7:  # 70% cache hit rate
            metrics.record_cache_hit()
        else:
            metrics.record_cache_miss()

        # Simulate delivery time
        delivery_time = random.uniform(0.1, 2.0)  # 0.1 to 2 seconds
        metrics.record_delivery_time(delivery_time)

        # Simulate engagement score
        engagement_score = random.uniform(1, 100)
        metrics.record_engagement_score(engagement_score)

        # Simulate request cost
        path = [0, 1, 2, 3]  # Example path through the network
        metrics.record_request_cost(path)

        # Simulate network congestion
        if random.random() < 0.05:  # 5% timeout rate
            metrics.record_timeout()

        # Simulate some processing time between requests
        time.sleep(0.01)

# Main execution
if __name__ == "__main__":
    metrics = PerformanceMetrics(num_servers=5)
    simulate_cdn_requests(metrics)

    summary = metrics.get_metrics_summary()
    print("Performance Metrics Summary:")
    for metric, value in summary.items():
        print(f"{metric}: {value}")
