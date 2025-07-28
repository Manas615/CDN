import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random
from matplotlib.animation import FuncAnimation
import pandas as pd

class AdaptiveCDN:
    def __init__(self, num_servers=10, cache_capacity=200, num_content=50):
        self.network = self.create_network(num_servers)
        self.cache = ViralLFUCache(cache_capacity)
        self.router = EngagementAwareRouter(self.network, self.cache)
        self.metrics = PerformanceMetrics(num_servers)
        self.data_gen = DataGenerator(num_content)
        self.content_paths = defaultdict(list)
        self.viral_threshold = 80
        self.num_content = num_content
        self.content_info = defaultdict(lambda: {'requests': 0, 'edge_requests': 0, 'delivery_times': []})
        
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(10, 15))
        self.bars = self.ax1.bar(range(num_content), self.data_gen.get_engagement_scores())
        self.ax1.axhline(y=self.viral_threshold, color='r', linestyle='--')
        self.ax1.set_ylim(0, 100)
        self.ax1.set_title('Content Engagement Scores')
        self.ax1.set_xlabel('Content ID')
        self.ax1.set_ylabel('Engagement Score')
        
        # Draw the network graph
        pos = nx.spring_layout(self.network)
        nx.draw(self.network, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_weight='bold', ax=self.ax3)
        self.ax3.set_title('Network Graph')

    def create_network(self, num_servers):
        G = nx.Graph()
        G.add_node('User')
        G.add_node('OriginServer')
        for i in range(num_servers):
            server_name = f'EdgeServer{i+1}'
            G.add_node(server_name)
            G.add_edge('User', server_name, latency=random.uniform(10, 50), bandwidth=random.uniform(100, 1000))
            G.add_edge(server_name, 'OriginServer', latency=random.uniform(50, 100), bandwidth=random.uniform(1000, 10000))
        return G

    def run_simulation(self, num_requests=2000):
        for _ in range(num_requests):
            content_id, engagement_score = self.data_gen.generate_request()
            path = self.router.route_content('User', content_id, engagement_score)
            self.process_request(content_id, engagement_score, path)
            self.content_paths[content_id].append(path)
            
            if _ % 10 == 0:
                self.update_plot()
                plt.pause(0.1)

        self.metrics.display_metrics()
        self.display_content_info()

    def process_request(self, content_id, engagement_score, path):
        is_available, location = self.cache.get(content_id)
        self.metrics.record_request(content_id, is_available)
        self.content_info[content_id]['requests'] += 1
        
        if is_available:
            self.metrics.record_cache_hit()
            if 'EdgeServer' in location:
                self.content_info[content_id]['edge_requests'] += 1
        else:
            self.metrics.record_cache_miss()
            if engagement_score > self.viral_threshold:
                print(f"Content {content_id} is predicted to go viral with an engagement score of {engagement_score}.")
                edge_server = self.find_nearest_edge_server(path)
                print(f"Caching content {content_id} at {edge_server}.")
                self.cache.put(content_id, engagement_score, edge_server)
            else:
                self.cache.put(content_id, engagement_score, 'OriginServer')

        delivery_time = self.calculate_delivery_time(path, location)
        self.metrics.record_delivery_time(delivery_time)
        self.metrics.record_engagement_score(engagement_score)
        self.content_info[content_id]['delivery_times'].append(delivery_time)

    def find_nearest_edge_server(self, path):
        return next((node for node in path if 'EdgeServer' in node), 'OriginServer')

    def calculate_delivery_time(self, path, cached_location):
        if cached_location in path:
            path = path[path.index(cached_location):]
        
        total_time = 0
        for i in range(len(path) - 1):
            edge_data = self.network[path[i]][path[i+1]]
            latency = edge_data['latency']
            bandwidth = edge_data['bandwidth']
            content_size = random.uniform(1, 10)  # MB
            transfer_time = content_size * 8 / bandwidth  # Convert MB to Mb
            total_time += latency + transfer_time
        return total_time

    def update_plot(self):
        engagement_scores = self.data_gen.get_engagement_scores()
        for bar, height in zip(self.bars, engagement_scores):
            bar.set_height(height)
        viral_content_count = len([c for c in engagement_scores if c > self.viral_threshold])
        self.ax2.clear()
        labels = ['Viral', 'Non-viral']
        sizes = [viral_content_count, len(engagement_scores) - viral_content_count]
        if sum(sizes) > 0:
            self.ax2.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        self.ax2.set_title('Viral vs Non-viral Content')

    def display_content_info(self):
        data = []
        for content_id, info in self.content_info.items():
            data.append({
                'Content ID': content_id,
                'Total Requests': info['requests'],
                'Edge Requests': info['edge_requests'],
                'Avg Delivery Time': np.mean(info['delivery_times']),
                'Is Viral': 'Yes' if self.data_gen.get_engagement_score(content_id) > self.viral_threshold else 'No'
            })
        df = pd.DataFrame(data)
        print("\nContent Information:")
        print(df.to_string(index=False))

class ViralLFUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.frequency = defaultdict(int)
        self.viral_threshold = 80
        self.viral_content = set()

    def get(self, key):
        if key in self.cache:
            self.frequency[key] += 1
            return True, self.cache[key]['location']
        return False, 'OriginServer'

    def put(self, key, engagement_score, location):
        if len(self.cache) >= self.capacity:
            lfu_key = min(self.frequency.keys(), key=self.frequency.get)
            if engagement_score > self.viral_threshold:
                del self.cache[lfu_key]
                del self.frequency[lfu_key]
            else:
                return
        self.cache[key] = {'score': engagement_score, 'location': location}
        self.frequency[key] += 1
        if engagement_score > self.viral_threshold:
            self.viral_content.add(key)

class EngagementAwareRouter:
    def __init__(self, network, cache):
        self.network = network
        self.cache = cache
        self.latency_weight = 0.5

    def route_content(self, source, content_id, engagement_score):
        paths = nx.single_source_dijkstra_path(self.network, source)
        is_cached, location = self.cache.get(content_id)
        if is_cached:
            return paths[location]
        best_path = min(paths.values(), key=lambda p: self.path_cost(p, engagement_score))
        return best_path

    def path_cost(self, path, engagement_score):
        latency = sum(self.network[path[i]][path[i+1]]['latency'] for i in range(len(path)-1))
        return self.latency_weight * latency + (1 - self.latency_weight) * (1 / engagement_score)

class PerformanceMetrics:
    def __init__(self, num_servers):
        self.server_usage = {i: {'cpu': [], 'memory': []} for i in range(num_servers)}
        self.cache_hits = 0
        self.cache_misses = 0
        self.delivery_times = []
        self.engagement_scores = []
        self.requests = defaultdict(int)
        self.availability = defaultdict(int)

    def record_request(self, content_id, is_available):
        self.requests[content_id] += 1
        if is_available:
            self.availability[content_id] += 1

    def record_cache_hit(self):
        self.cache_hits += 1

    def record_cache_miss(self):
        self.cache_misses += 1

    def record_delivery_time(self, time):
        self.delivery_times.append(time)

    def record_engagement_score(self, score):
        self.engagement_scores.append(score)

    def display_metrics(self):
        print(f"Cache Hits: {self.cache_hits}")
        print(f"Cache Misses: {self.cache_misses}")
        print(f"Cache Hit Percentage: {self.cache_hits / (self.cache_hits + self.cache_misses) * 100:.2f}%")
        print(f"Average Delivery Time: {np.mean(self.delivery_times):.2f} seconds")
        print(f"Average Engagement Score: {np.mean(self.engagement_scores):.2f}")
        
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(self.delivery_times)
        plt.title("Content Delivery Times")
        plt.xlabel("Request Number")
        plt.ylabel("Delivery Time (seconds)")
        plt.subplot(2, 2, 2)
        plt.plot(self.engagement_scores)
        plt.title("Engagement Scores")
        plt.xlabel("Request Number")
        plt.ylabel("Engagement Score")
        plt.subplot(2, 2, 3)
        plt.bar(['Hits', 'Misses'], [self.cache_hits, self.cache_misses])
        plt.title("Cache Performance")
        plt.subplot(2, 2, 4)
        plt.bar(list(self.requests.keys()), list(self.requests.values()))
        plt.title("Content Request Frequency")
        plt.xlabel("Content ID")
        plt.ylabel("Number of Requests")
        plt.tight_layout()
        plt.show()

class DataGenerator:
    def __init__(self, num_content=50):
        self.num_content = num_content
        self.content_data = self.generate_content_data()

    def generate_content_data(self):
        return {i: {
            'engagement_score': random.uniform(10, 100),
            'size': random.randint(1, 10)  # MB
        } for i in range(self.num_content)}

    def generate_request(self):
        content_id = random.randint(0, self.num_content - 1)
        self.content_data[content_id]['engagement_score'] *= random.uniform(0.9, 1.1)
        self.content_data[content_id]['engagement_score'] = min(100, max(10, self.content_data[content_id]['engagement_score']))
        return content_id, self.content_data[content_id]['engagement_score']

    def get_engagement_scores(self):
        return [self.content_data[i]['engagement_score'] for i in range(self.num_content)]

    def get_engagement_score(self, content_id):
        return self.content_data[content_id]['engagement_score']

if __name__ == "__main__":
    cdn_simulation = AdaptiveCDN(num_servers=10, cache_capacity=200, num_content=50)
    cdn_simulation.run_simulation(num_requests=2000)
