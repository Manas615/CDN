import networkx as nx
import random

class EngagementAwareRouter:
    def __init__(self, network):
        self.network = network
        self.latency_weight = 0.5  # Initial weight, balanced between latency and engagement

    def update_network_conditions(self):
        # Simulate changing network conditions
        for edge in self.network.edges():
            self.network[edge[0]][edge[1]]['latency'] = random.uniform(10, 100)
        
        # Update latency weight based on network congestion
        avg_latency = sum(d['latency'] for _, _, d in self.network.edges(data=True)) / self.network.number_of_edges()
        self.latency_weight = min(1.0, max(0.1, avg_latency / 100))

    def calculate_path_cost(self, path, content_id):
        total_latency = sum(self.network[path[i]][path[i+1]]['latency'] for i in range(len(path)-1))
        end_server = path[-1]
        engagement_score = self.network.nodes[end_server]['cached_content'].get(content_id, 1)
        
        return self.latency_weight * total_latency + (1 - self.latency_weight) * (1 / engagement_score)

    def find_best_path(self, source, content_id):
        all_paths = nx.all_simple_paths(self.network, source=source, target='OriginServer')
        best_path = min(all_paths, key=lambda path: self.calculate_path_cost(path, content_id))
        return best_path

    def route_content(self, source, content_id):
        self.update_network_conditions()
        best_path = self.find_best_path(source, content_id)
        return best_path

# Example usage
def create_sample_network():
    G = nx.Graph()
    G.add_node('User')
    G.add_node('EdgeServer1', cached_content={'content1': 50, 'content2': 80})
    G.add_node('EdgeServer2', cached_content={'content1': 30, 'content3': 70})
    G.add_node('OriginServer', cached_content={'content1': 10, 'content2': 20, 'content3': 30})
    
    G.add_edge('User', 'EdgeServer1', latency=20)
    G.add_edge('User', 'EdgeServer2', latency=30)
    G.add_edge('EdgeServer1', 'OriginServer', latency=50)
    G.add_edge('EdgeServer2', 'OriginServer', latency=40)
    
    return G

# Main execution
if __name__ == "__main__":
    network = create_sample_network()
    router = EngagementAwareRouter(network)

    # Simulate routing for different content
    for content in ['content1', 'content2', 'content3']:
        path = router.route_content('User', content)
        print(f"Best path for {content}: {' -> '.join(path)}")
        print(f"Current Latency Weight: {router.latency_weight:.2f}")
        print("---")
