import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import simpy  # For Discrete Event Simulation

class CDNEnvironment:
    """
    Simulates a Content Delivery Network (CDN) environment for social media content.
    Includes network topology, edge servers with caching, user requests, and
    engagement modeling.
    """

    def __init__(self, num_edge_servers=3, origin_server_capacity=1000,  network_stability=0.95):
        """
        Initializes the CDN environment.

        Args:
            num_edge_servers (int): Number of edge servers in the CDN.
            origin_server_capacity (int): Storage capacity of the origin server.
            network_stability (float): Probability of network connection staying active.
        """
        self.num_edge_servers = num_edge_servers
        self.origin_server_capacity = origin_server_capacity
        self.network_stability = network_stability
        self.network = nx.Graph()  # Network topology
        self.edge_server_names = [] # Stores the names of edge servers

        self.content_catalog = {}  # Content metadata (size, engagement)

        self.env = simpy.Environment() # Simulation environment

        self.setup_network()
        self.generate_initial_content()

    def setup_network(self):
        """
        Creates the network topology with an origin server and edge servers.
        Connects the User to edge servers with varying latency.
        """
        self.network.add_node("User")
        self.network.add_node("OriginServer", capacity=self.origin_server_capacity, cached_content={})

        # Create edge servers
        for i in range(self.num_edge_servers):
            server_name = f"EdgeServer{i+1}"
            self.edge_server_names.append(server_name) # Store the server name
            self.network.add_node(server_name, capacity=100, cached_content={})  # Edge server with limited capacity
            # Connect User to edge servers with random latencies
            self.network.add_edge("User", server_name, latency=random.randint(10, 30))  # Example latencies

            #Connect edge servers to OriginServer
            self.network.add_edge(server_name, "OriginServer", latency = random.randint(40,60))

    def generate_initial_content(self, num_content=5):
        """
        Generates a set of initial content with random sizes and engagement scores.
        """
        for i in range(num_content):
            content_id = f"content{i+1}"
            self.content_catalog[content_id] = {
                "size": random.randint(10, 50),  # Content size (example: MB)
                "engagement": random.randint(10, 100), # Initial engagement score
                "popularity_trend": self.generate_popularity_trend(),  # Function to generate popularity trend
            }

    def generate_popularity_trend(self):
        """
        Simulates a popularity trend for content (e.g., viral, steady, declining).
        This is a very simplified example. You can make this more sophisticated.
        """
        trend_type = random.choice(["viral", "steady", "declining"])
        if trend_type == "viral":
            return lambda x: min(1000, int(x**2))  # Exponential growth, capped at 1000
        elif trend_type == "steady":
            return lambda x: 50 + random.randint(-10, 10) # Roughly constant
        else: # declining
            return lambda x: max(0, 100 - int(x**1.5)) # Decreasing popularity
    def simulate_user_request(self, content_id):
        """
        Simulates a user requesting specific content.
        """
        yield self.env.timeout(random.randint(1,5)) # Simulate some delay before the request

        print(f"{self.env.now}: User requesting content {content_id}")
        best_server = self.determine_best_server(content_id)

        if best_server:
             print(f"{self.env.now}: Serving {content_id} from {best_server}")
             #Update content access counts in best_server
             if content_id in self.network.nodes[best_server]["cached_content"]:
                self.network.nodes[best_server]["cached_content"][content_id]+= 1
             else:
                 self.network.nodes[best_server]["cached_content"][content_id] = 1

        else:
            print(f"{self.env.now}: Content {content_id} not available.")

    def determine_best_server(self, content_id):
        """
        Determines the best server to serve content based on latency and cache status.

        This example prioritizes edge servers with cached content.
        """
        available_servers = []

        for server_name in self.edge_server_names: # Use the stored list
            if content_id in self.network.nodes[server_name]["cached_content"]:
                available_servers.append(server_name)

        if available_servers:
            #Find the edge server with the minimum latency
            best_server = min(available_servers, key = lambda server: self.network["User"][server]["latency"])
            return best_server
        else:
            #Content is not cached in any edge server, so fetch content from the Origin Server
            #Check if content is present in the Origin Server
            if(content_id in self.network.nodes["OriginServer"]["cached_content"]):
                #Fetch from origin Server and cache to edge server
                return "OriginServer"
            else:
                #Cannot find best_server
                return None

    def cache_content_on_edge(self, edge_server, content_id):
        """
        Caches content on a given edge server, evicting other content if necessary.

        Args:
            edge_server (str): Name of the edge server.
            content_id (str): ID of the content to cache.
        """
        edge_node = self.network.nodes[edge_server]
        content_size = self.content_catalog[content_id]["size"]
        #Check if the edge_server is capable of caching
        if (sum(edge_node["cached_content"].values()) + content_size) <= edge_node["capacity"]:
            edge_node["cached_content"][content_id] = content_size

        else:
            print(f"{edge_server} is full, content not cached")

    def simulate(self, simulation_time=100):
         """
         Runs the simulation for a specified time.

         Args:
             simulation_time (int): The duration of the simulation.
         """

         #Start the simulation
         self.env.process(self.simulate_content_requests())
         self.env.run(until=simulation_time)

    def simulate_content_requests(self):
        """
        Generates user requests for content over time.
        """
        content_ids = list(self.content_catalog.keys()) #Get the available content
        while True:
            content_id = random.choice(content_ids)  # Randomly select content
            self.env.process(self.simulate_user_request(content_id)) #Process content request

            yield self.env.timeout(random.randint(1, 5)) # Wait before next request
    def visualize_network(self):
        """
        Visualizes the network topology with edge latencies.
        """
        pos = nx.spring_layout(self.network)  # Layout algorithm for node positioning
        nx.draw(self.network, pos, with_labels=True, node_size=2000, node_color="skyblue", font_size=10)

        # Add edge labels for latency
        edge_labels = {(u, v): data['latency'] for u, v, data in self.network.edges(data=True)}
        nx.draw_networkx_edge_labels(self.network, pos, edge_labels=edge_labels, font_size=8)

        plt.title("CDN Network Topology")
        plt.show()
# --- Example Usage ---
# Create the CDN environment
cdn_env = CDNEnvironment(num_edge_servers=2)

#Run the simulation
cdn_env.simulate(simulation_time=50)

#Visualize the network
cdn_env.visualize_network()

#Print the final cache state
for server_name in cdn_env.edge_server_names:
    print(f"{server_name} cached content: {cdn_env.network.nodes[server_name]['cached_content']}")
