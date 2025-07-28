# -*- coding: utf-8 -*-
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation
import matplotlib.style as style
from collections import defaultdict, deque
import random
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional, Set
import json

# Apply a modern style
style.use('seaborn-v0_8-darkgrid')

class ViralLFUCache:
    
    def __init__(self, capacity: int, viral_threshold: float = 80.0):
       
        self.capacity: int = capacity
        self.cache: Dict[int, Dict[str, Any]] = {}  # {content_id: {'score': float, 'location': str, 'size': int}}
        self.frequency: Dict[int, int] = defaultdict(int)
        self.viral_threshold: float = viral_threshold
        self.viral_content_in_cache: Set[int] = set()
        print(f"Cache initialized with capacity: {capacity}")

    def get(self, key: int) -> Tuple[bool, Optional[str]]:
       
        if key in self.cache:
            self.frequency[key] += 1
            # print(f"Cache GET: Hit for content {key} at {self.cache[key]['location']}. Freq: {self.frequency[key]}")
            return True, self.cache[key]['location']
        # print(f"Cache GET: Miss for content {key}")
        return False, None

    def put(self, key: int, engagement_score: float, location: str, size: int) -> Optional[int]:
        
        evicted_key = None
        if key in self.cache:
            # Update existing entry's score and location if necessary, increment freq
            self.cache[key]['score'] = engagement_score
            self.cache[key]['location'] = location # Location might change if re-cached closer
            self.frequency[key] += 1
            if engagement_score > self.viral_threshold:
                self.viral_content_in_cache.add(key)
            else:
                self.viral_content_in_cache.discard(key)
            # print(f"Cache PUT: Updated content {key}. Freq: {self.frequency[key]}")
            return evicted_key # No eviction on update

        if len(self.cache) >= self.capacity:
            # Eviction needed
            lfu_key = min(self.frequency, key=self.frequency.get)

            # Simple viral preference: Don't evict viral content if possible
            # If LFU is viral, try finding a non-viral LFU
            if lfu_key in self.viral_content_in_cache:
                non_viral_lfu_candidates = {k: v for k, v in self.frequency.items() if k not in self.viral_content_in_cache}
                if non_viral_lfu_candidates:
                     lfu_key = min(non_viral_lfu_candidates, key=non_viral_lfu_candidates.get)
                # If all are viral, we have to evict the LFU viral one

            # Perform eviction
            print(f"Cache PUT: Evicting content {lfu_key} (Freq: {self.frequency[lfu_key]}) to make space for {key}.")
            del self.cache[lfu_key]
            del self.frequency[lfu_key]
            self.viral_content_in_cache.discard(lfu_key)
            evicted_key = lfu_key

        # Add the new item
        self.cache[key] = {'score': engagement_score, 'location': location, 'size': size}
        self.frequency[key] = 1 # Start frequency at 1
        if engagement_score > self.viral_threshold:
            self.viral_content_in_cache.add(key)
        print(f"Cache PUT: Added content {key} at {location}. Freq: {self.frequency[key]}. Cache size: {len(self.cache)}/{self.capacity}")
        return evicted_key

    def get_cache_status(self) -> Dict[str, Any]:
         """Returns current cache size and viral content count."""
         return {"size": len(self.cache), "viral_count": len(self.viral_content_in_cache)}


class EngagementAwareRouter:
   
    def __init__(self, network: nx.Graph, cache: ViralLFUCache, congestion_penalty: float = 50.0):
        
        self.network: nx.Graph = network
        self.cache: ViralLFUCache = cache
        self.congestion_penalty: float = congestion_penalty
        print(f"Router initialized with congestion penalty: {congestion_penalty}")

    def _congestion_aware_weight(self, u, v, data) -> float:
        """Calculate edge weight considering latency and congestion."""
        latency = data.get('latency', float('inf')) # Base latency
        load = data.get('load', 0.0)
        capacity = data.get('capacity', 1.0) # Avoid division by zero if capacity is missing or zero

        if capacity <= 0:
            # Handle zero or negative capacity - treat as infinitely congested if loaded
            return float('inf') if load > 0 else latency

        congestion_cost = self.congestion_penalty * (load / capacity)

        # Return combined weight, ensuring it's at least the base latency
        return latency + max(0, congestion_cost)

    def _calculate_path_latency(self, path: List[str]) -> float:
        """Calculates the total *base* latency for a given path (ignoring load)."""
        latency = 0
        for i in range(len(path) - 1):
            try:
                latency += self.network[path[i]][path[i+1]]['latency']
            except KeyError:
                print(f"Warning: Edge not found between {path[i]} and {path[i+1]}")
                return float('inf') # Penalize missing edges heavily
        return latency

    def route_content(self, source: str, content_id: int, engagement_score: float) -> Tuple[List[str], str]:
        
        is_cached, cache_location = self.cache.get(content_id)

        # Define target: either cache location or origin server
        target_node = 'OriginServer'
        if is_cached and cache_location:
             target_node = cache_location

        try:
            # Calculate shortest path using the dynamic congestion-aware weight
            path = nx.dijkstra_path(self.network, source, target_node, weight=self._congestion_aware_weight)

            # Determine the actual source based on where the path ended
            actual_source_location = target_node if target_node != 'OriginServer' else 'OriginServer'

            # If cached, but path to cache failed (e.g., due to extreme congestion), retry to Origin
            if is_cached and cache_location and actual_source_location == 'OriginServer':
                 print(f"Warning: Path found to Origin instead of cache location {cache_location} for content {content_id}. Routing to Origin.")
                 path = nx.dijkstra_path(self.network, source, 'OriginServer', weight=self._congestion_aware_weight)
                 actual_source_location = 'OriginServer'


            # print(f"Routing (Congestion-Aware): Content {content_id} from {source} to {actual_source_location}. Path: {' -> '.join(path)}")
            return path, actual_source_location

        except nx.NetworkXNoPath:
             # If no path even to origin exists (network partition or extreme congestion)
             print(f"Error: No congestion-aware path found from {source} to {target_node} for content {content_id}!")
             # Try finding path with just latency as fallback? Or just fail.
             try:
                 path_latency_only = nx.dijkstra_path(self.network, source, target_node, weight='latency')
                 print(f"Fallback: Routing via latency-only path: {' -> '.join(path_latency_only)}")
                 actual_source_location = target_node if target_node != 'OriginServer' else 'OriginServer'
                 return path_latency_only, actual_source_location
             except nx.NetworkXNoPath:
                  print(f"FATAL: No path found from {source} to {target_node} even with latency only.")
                  return [], 'OriginServer' # Return empty path indicates failure

class PerformanceMetrics:
    """
    Tracks and calculates various performance metrics for the CDN simulation.
    """
    def __init__(self):
        """Initializes metric storage."""
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.delivery_times: List[float] = []
        self.engagement_scores: List[float] = []
        self.requests_per_content: Dict[int, int] = defaultdict(int)
        self.edge_hits_per_content: Dict[int, int] = defaultdict(int)
        print("PerformanceMetrics initialized.")

    def record_request(self, content_id: int, is_cache_hit: bool, is_edge_hit: bool):
        """Records a content request and whether it was a cache/edge hit."""
        self.requests_per_content[content_id] += 1
        if is_cache_hit:
            self.cache_hits += 1
            if is_edge_hit:
                 self.edge_hits_per_content[content_id] += 1
        else:
            self.cache_misses += 1

    def record_delivery_time(self, time: float):
        """Records the delivery time for a request."""
        if time is not None and time < float('inf'): # Ignore failed deliveries
             self.delivery_times.append(time)

    def record_engagement_score(self, score: float):
        """Records the engagement score for a request."""
        self.engagement_scores.append(score)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Calculates and returns a summary of current metrics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        avg_delivery_time = np.mean(self.delivery_times) if self.delivery_times else 0
        avg_engagement = np.mean(self.engagement_scores) if self.engagement_scores else 0

        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "avg_delivery_time": avg_delivery_time,
            "avg_engagement": avg_engagement,
            "total_requests": total_requests
        }

    def display_final_metrics(self):
        """Prints final metrics and generates summary plots."""
        summary = self.get_metrics_summary()
        print("\n--- Simulation Final Metrics ---")
        print(f"Total Requests Processed: {summary['total_requests']}")
        print(f"Cache Hits: {summary['cache_hits']}")
        print(f"Cache Misses: {summary['cache_misses']}")
        print(f"Cache Hit Percentage: {summary['hit_rate']:.2f}%")
        print(f"Average Delivery Time: {summary['avg_delivery_time']:.3f} ms") # Assuming time is in ms
        print(f"Average Engagement Score: {summary['avg_engagement']:.2f}")

        if not self.delivery_times:
            print("No successful deliveries recorded.")
            return

        plt.figure(figsize=(12, 10))
        plt.suptitle("CDN Simulation - Final Performance Summary", fontsize=16)

        # Plot 1: Delivery Times Over Simulation
        plt.subplot(2, 2, 1)
        plt.plot(self.delivery_times, label='Delivery Time', alpha=0.7)
        # Add a moving average
        if len(self.delivery_times) >= 50:
             moving_avg = pd.Series(self.delivery_times).rolling(window=50).mean()
             plt.plot(moving_avg, label='50-req Moving Avg', color='red', linewidth=2)
        plt.title("Content Delivery Times")
        plt.xlabel("Request Number")
        plt.ylabel("Delivery Time (ms)")
        plt.legend()
        plt.grid(True)

        # Plot 2: Engagement Scores Over Simulation
        plt.subplot(2, 2, 2)
        plt.plot(self.engagement_scores, label='Engagement Score', alpha=0.7, color='green')
        plt.title("Engagement Scores of Requested Content")
        plt.xlabel("Request Number")
        plt.ylabel("Engagement Score")
        plt.grid(True)

        # Plot 3: Cache Performance (Pie Chart)
        plt.subplot(2, 2, 3)
        if summary['total_requests'] > 0:
            labels = ['Cache Hits', 'Cache Misses']
            sizes = [summary['cache_hits'], summary['cache_misses']]
            colors = ['#99ff99', '#ff9999']
            explode = (0.1, 0) # explode 1st slice
            plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                    shadow=True, startangle=140)
            plt.title("Cache Performance")
        else:
             plt.text(0.5, 0.5, "No requests", ha='center', va='center')
             plt.title("Cache Performance")


        # Plot 4: Content Request Frequency
        plt.subplot(2, 2, 4)
        if self.requests_per_content:
            sorted_requests = sorted(self.requests_per_content.items())
            content_ids = [item[0] for item in sorted_requests]
            counts = [item[1] for item in sorted_requests]
            plt.bar(range(len(content_ids)), counts, tick_label=[str(cid) for cid in content_ids])
            plt.title("Content Request Frequency")
            plt.xlabel("Content ID")
            plt.ylabel("Number of Requests")
            plt.xticks(rotation=90, fontsize=8)
        else:
             plt.text(0.5, 0.5, "No requests", ha='center', va='center')
             plt.title("Content Request Frequency")


        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
        plt.show()


class DataGenerator:
    """
    Generates content data and simulates user requests with fluctuating engagement scores.
    Loads initial data from a JSON file.
    """
    def __init__(self, data_file_path: str):
        """
        Initializes the data generator by loading data from a file.

        Args:
            data_file_path (str): Path to the JSON file containing initial content data.
        """
        self.content_data: Dict[int, Dict[str, Any]] = self._load_content_data_from_file(data_file_path)
        self.num_content: int = len(self.content_data)
        print(f"DataGenerator initialized with {self.num_content} content items from {data_file_path}.")

    def _load_content_data_from_file(self, file_path: str) -> Dict[int, Dict[str, Any]]:
        """Loads initial content properties from a JSON file."""
        data = {}
        try:
            with open(file_path, 'r') as f:
                loaded_data = json.load(f)
            for item in loaded_data:
                content_id = item['content_id']
                score = item['current_score']
                size = item['size_mb'] # Use size_mb from JSON
                # Store initial score for fluctuation logic, and use correct size key
                data[content_id] = {
                    'initial_score': score, # Use loaded score as initial
                    'current_score': score,
                    'size': size
                }
            print(f"Successfully loaded data for {len(data)} items.")
        except FileNotFoundError:
            print(f"Error: Data file not found at {file_path}. Exiting.")
            exit()
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error reading or parsing data file {file_path}: {e}. Exiting.")
            exit()
        return data

    def _generate_initial_content_data(self) -> Dict[int, Dict[str, Any]]:
        """Generates initial properties for each content item."""
        # This method is no longer used, replaced by _load_content_data_from_file
        # Kept here temporarily or could be removed
        # ... (previous implementation removed) ...
        pass # Or raise NotImplementedError


    def generate_request(self) -> Tuple[str, int, float, int]:
        
        scores = np.array([self.content_data[i]['current_score'] for i in sorted(self.content_data.keys())])
        probabilities = scores / scores.sum() if scores.sum() > 0 else None
        # Use range(self.num_content) assuming IDs are 0 to N-1 and contiguous
        content_id = np.random.choice(list(self.content_data.keys()), p=probabilities)

        # 3. Simulate score fluctuation
        current_score = self.content_data[content_id]['current_score']
        initial_score = self.content_data[content_id]['initial_score']
        change = (initial_score - current_score) * 0.05 + random.uniform(-3, 3)
        new_score = current_score + change
        if random.random() < 0.01:
             new_score *= random.uniform(1.1, 1.5)
        self.content_data[content_id]['current_score'] = min(100, max(1, new_score))

        size = self.content_data[content_id]['size']
        # Return content info; user selection happens in AdaptiveCDN
        return content_id, self.content_data[content_id]['current_score'], size

    def get_all_current_scores(self) -> List[float]:
        """Returns a list of the current engagement scores for all content."""
        # Ensure iteration is over the dictionary keys (content IDs)
        return [self.content_data[i]['current_score'] for i in sorted(self.content_data.keys())]

    def get_content_info(self, content_id: int) -> Dict[str, Any]:
         """ Returns info for a specific content ID """
         # Return a copy to prevent accidental modification
         return self.content_data.get(content_id, {}).copy()


class AdaptiveCDN:
   
    def __init__(self, num_servers: int = 10, num_users: int = 5, cache_capacity: int = 100, viral_threshold: float = 80.0, data_file: str = 'initial_content_data.json', load_decay_factor: float = 0.1, congestion_penalty: float = 50.0):
        
        print("Initializing Adaptive CDN Simulation...")
        self.num_servers: int = num_servers
        self.num_users: int = num_users # Store num_users
        self.viral_threshold: float = viral_threshold
        self.load_decay_factor = max(0, min(1, load_decay_factor)) # Ensure 0 <= factor <= 1

        # Instantiate DataGenerator first using the data file
        self.data_gen: DataGenerator = DataGenerator(data_file)
        self.num_content: int = self.data_gen.num_content

        # Create network (self.user_nodes is set inside _create_network)
        self.network: nx.Graph = self._create_network(num_servers, num_users)
        # self.network_pos is also set inside _create_network now

        # Cache
        self.cache: ViralLFUCache = ViralLFUCache(cache_capacity, viral_threshold)

        # Router needs the network with load attributes and penalty factor
        self.router: EngagementAwareRouter = EngagementAwareRouter(self.network, self.cache, congestion_penalty)
        self.metrics: PerformanceMetrics = PerformanceMetrics()

        # Simulation state variables
        self.current_step: int = 0
        self.last_request_info: Dict = {}
        self.edge_server_flash: Dict[str, int] = defaultdict(int)

        # --- Visualization Setup ---
        self.fig = plt.figure(figsize=(16, 9))
        gs = GridSpec(2, 2, figure=self.fig)
        self.ax_network = self.fig.add_subplot(gs[0, 0])
        self.ax_engagement = self.fig.add_subplot(gs[0, 1])
        self.ax_delivery = self.fig.add_subplot(gs[1, 0])
        self.ax_cache = self.fig.add_subplot(gs[1, 1])

        # Placeholder for engagement bars (will be created in first update)
        self.engagement_bars = None

        # Delivery time plot setup (using deque for efficiency)
        self.delivery_time_history = deque(maxlen=100) # Show last 100 delivery times
        self.delivery_line, = self.ax_delivery.plot([], [], lw=2, label='Delivery Time (ms)')
        self.ax_delivery.set_xlim(0, 100)
        self.ax_delivery.set_ylim(0, 200) # Initial guess, will adapt
        self.ax_delivery.set_title('Recent Delivery Times')
        self.ax_delivery.set_xlabel('Recent Requests')
        self.ax_delivery.set_ylabel('Time (ms)')
        self.ax_delivery.legend(loc='upper left')
        self.ax_delivery.grid(True)

        # Cache performance bars setup
        self.cache_bars = None
        self.ax_cache.set_xticks([0, 1])
        self.ax_cache.set_xticklabels(['Hits', 'Misses'])
        self.ax_cache.set_title('Cache Performance (Cumulative)')
        self.ax_cache.set_ylabel('Count')
        self.ax_cache.grid(axis='y')

        print("Visualization setup complete.")

    def _create_network(self, num_servers: int, num_users: int = 5) -> nx.Graph:
        
        G = nx.Graph()
        # Add Origin Server
        G.add_node('OriginServer', type='origin', pos=(0.5, 1), 
                   label='Origin\nServer') # Added label for display

        # Add Edge Servers
        edge_servers = [f'EdgeServer{i+1}' for i in range(num_servers)]
        edge_pos = {}
        # Arrange edge servers in a circle/ellipse below the origin
        for i, server_name in enumerate(edge_servers):
            angle = 2 * np.pi * i / num_servers
            x = 0.5 + 0.4 * np.cos(angle)
            y = 0.6 + 0.2 * np.sin(angle) # Ellipse shape
            edge_pos[server_name] = (x, y)
            G.add_node(server_name, type='edge', 
                      label=f'Edge\n{i+1}') # Shortened label for cleaner display
            # Edge to Origin links
            bw = random.uniform(500, 2000) # Mbps
            G.add_edge(server_name, 'OriginServer',
                       latency=random.uniform(50, 150), # ms
                       bandwidth=bw, # Mbps
                       capacity=bw, # Initial capacity equals bandwidth
                       load=0.0) # Current load starts at 0

        # Add User Nodes
        self.user_nodes = [f'User{i+1}' for i in range(num_users)]
        user_pos = {}
        # Arrange user nodes in a wider circle/ellipse at the bottom
        for i, user_name in enumerate(self.user_nodes):
            angle = 2 * np.pi * i / num_users
            x = 0.5 + 0.6 * np.cos(angle + np.pi / num_users) # Offset angle slightly
            y = 0.1 + 0.1 * np.sin(angle + np.pi / num_users)
            user_pos[user_name] = (x, y)
            G.add_node(user_name, type='user', 
                      label=f'User {i+1}') # User display label

            # Connect each user to a subset of edge servers (e.g., 2-4 closest)
            # Instead of closest, let's connect randomly for simplicity now
            num_user_connections = random.randint(2, max(2, num_servers // 2))
            connected_edges = random.sample(edge_servers, num_user_connections)
            for server_name in connected_edges:
                bw = random.uniform(50, 500) # Mbps
                G.add_edge(user_name, server_name,
                           latency=random.uniform(5, 50), # ms, lower latency user->edge
                           bandwidth=bw, # Mbps
                           capacity=bw,
                           load=0.0)

        # Add some inter-edge links
        num_inter_edge_links = num_servers # Increase potential paths
        for _ in range(num_inter_edge_links):
             s1, s2 = random.sample(edge_servers, 2)
             if not G.has_edge(s1, s2):
                 bw = random.uniform(100, 1000) # Mbps
                 G.add_edge(s1, s2,
                            latency=random.uniform(10, 40), # ms
                            bandwidth=bw, # Mbps
                            capacity=bw,
                            load=0.0)

        # Store positions for drawing
        self.network_pos = {'OriginServer': (0.5, 1)}
        self.network_pos.update(edge_pos)
        self.network_pos.update(user_pos)
        # Check connectivity - ensure all users can reach the origin
        # (Not strictly necessary with random links, but good practice)
        for user in self.user_nodes:
            if not nx.has_path(G, user, 'OriginServer'):
                print(f"Warning: User {user} may not have a path to OriginServer. Check network topology.")

        print(f"Network created with {num_users} Users, 1 Origin, {num_servers} Edge Servers.")
        return G

    def _decay_load(self):
        
        decay_multiplier = 1.0 - self.load_decay_factor
        for u, v, data in self.network.edges(data=True):
            if 'load' in data:
                data['load'] *= decay_multiplier
                # Ensure load doesn't become insignificantly small (optional)
                if data['load'] < 1e-3:
                    data['load'] = 0.0

    def _calculate_delivery_time(self, path: List[str], content_size_mb: int) -> Optional[float]:
        
        if not path or len(path) < 2:
             print("Warning: Invalid path for time calculation.")
             return None

        total_time_ms = 0.0
        content_size_mbit = content_size_mb * 8 # Convert MB to Megabits

        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            try:
                edge_data = self.network[u][v]
                latency_ms = edge_data['latency']
                bandwidth_mbps = edge_data['bandwidth']

                if bandwidth_mbps <= 0:
                     print(f"Warning: Zero or negative bandwidth on edge {u}-{v}. Skipping transfer time.")
                     transfer_time_ms = 0
                else:
                     transfer_time_seconds = content_size_mbit / bandwidth_mbps
                     transfer_time_ms = transfer_time_seconds * 1000

                total_time_ms += latency_ms + transfer_time_ms
                # print(f"  Hop {u}->{v}: Latency={latency_ms:.2f}ms, BW={bandwidth_mbps}Mbps, Size={content_size_mb}MB, Transfer={transfer_time_ms:.2f}ms")

            except KeyError:
                print(f"Error: Edge data not found for {u}-{v} in path. Cannot calculate time.")
                return float('inf') # Indicate failure/impossibly long time

        # print(f"Calculated Delivery Time: {total_time_ms:.2f} ms for path {' -> '.join(path)}")
        return total_time_ms

    def _process_request(self, user_node: str, content_id: int, engagement_score: float, content_size: int) -> None:
        
        path, source_location = self.router.route_content(user_node, content_id, engagement_score)

        if not path:
             print(f"Request failed: No path found for content {content_id} from {user_node}.")
             self.metrics.record_request(content_id, is_cache_hit=False, is_edge_hit=False) # Record as miss
             self.last_request_info = {'user': user_node, 'path': [], 'cached_at': None, 'delivery_time': None, 'viral_cache_event': False}
             return # Skip processing if no path

        # 2. Update Load on the path edges
        # Increment load by content size (simplistic model)
        load_increment = content_size # MB
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            if self.network.has_edge(u, v):
                self.network[u][v]['load'] += load_increment
                # print(f"  Increased load on {u}-{v} by {load_increment}. New load: {self.network[u][v]['load']:.1f}")
            else:
                print(f"Warning: Edge {u}-{v} from path not found in network during load update.")


        # 3. Determine if it was a cache hit and where
        is_cache_hit = (source_location != 'OriginServer')
        is_edge_hit = is_cache_hit # Cached means edge hit in this model

        # 4. Calculate delivery time (using base bandwidth/latency for now)
        delivery_time = self._calculate_delivery_time(path, content_size)

        # 5. Record metrics
        self.metrics.record_request(content_id, is_cache_hit, is_edge_hit)
        if delivery_time is not None:
             self.metrics.record_delivery_time(delivery_time)
             self.delivery_time_history.append(delivery_time) # Add to deque for plotting
        self.metrics.record_engagement_score(engagement_score)

        # 6. Cache Management (Decision to cache/update after delivery)
        viral_cache_event = False
        if not is_cache_hit:
             # Cache Miss: Decide if and where to cache it now
             if engagement_score > self.viral_threshold:
                 # --- MODIFIED: Cache viral content at a *random* edge server connected to the user ---
                 user_neighbors = list(self.network.neighbors(user_node))
                 edge_server_neighbors = [n for n in user_neighbors if self.network.nodes[n].get('type') == 'edge']

                 if edge_server_neighbors:
                     target_cache_server = random.choice(edge_server_neighbors)
                     print(f"Viral Action: Content {content_id} (Score: {engagement_score:.1f}) is viral. Caching at random neighbor {target_cache_server} for user {user_node}.")
                     self.cache.put(content_id, engagement_score, target_cache_server, content_size)
                     self.edge_server_flash[target_cache_server] = 3 # Flash for 3 frames
                     viral_cache_event = True
                 else:
                     # Fallback if user has no direct edge connections (shouldn't happen with current _create_network)
                     edge_server_in_path = next((node for node in reversed(path) if 'EdgeServer' in node), None)
                     if edge_server_in_path:
                          print(f"Viral Action (Fallback): User {user_node} has no edge neighbors. Caching at edge server in path: {edge_server_in_path}.")
                          self.cache.put(content_id, engagement_score, edge_server_in_path, content_size)
                          self.edge_server_flash[edge_server_in_path] = 3
                          viral_cache_event = True
                     else:
                          print(f"Viral Action (Fallback Failed): Content {content_id} is viral, but no edge server found for user {user_node} or in path {path}. Caching logically at Origin.")
                          self.cache.put(content_id, engagement_score, 'OriginServer', content_size)
             else:
                 
                 pass
        else:
             # Cache Hit: Update score/freq
             self.cache.put(content_id, engagement_score, source_location, content_size)

        # Store info for plotting
        self.last_request_info = {
            'user': user_node, # Store requesting user
            'path': path,
            'cached_at': source_location if is_cache_hit else None,
            'delivery_time': delivery_time,
            'viral_cache_event': viral_cache_event
            }

    def _update_plot(self, frame: int) -> Tuple:
        """Update function for Matplotlib animation."""
        # --- Apply Load Decay ---
        self._decay_load()

        # --- Run one simulation step ---
        if self.current_step < self.num_requests:
             # 1. Select requesting user randomly
             requesting_user = random.choice(self.user_nodes)

             # 2. Generate content request details
             content_id, engagement_score, content_size = self.data_gen.generate_request() # No user needed here anymore

             # 3. Process the request for the selected user
             self._process_request(requesting_user, content_id, engagement_score, content_size)

             self.current_step += 1
        else:
             # Stop animation if simulation ends
             # self.anim.event_source.stop() # Might cause issues if stopped too early
             pass

        # --- Update Network Plot (ax_network) ---
        self.ax_network.cla() # Clear previous drawing
        
        # Create a custom visualization with more information
        
        # 1. Draw edges first (so they're below nodes)
        edge_list = list(self.network.edges(data=True))
        
        # Prepare data structures for edge styling
        edge_colors = []
        edge_widths = []
        edge_styles = []
        edge_alphas = []
        
        # Identify path edges for highlighting
        last_path = self.last_request_info.get('path', [])
        path_edge_set = set()
        if last_path:
            path_edge_set = {(last_path[i], last_path[i+1]) for i in range(len(last_path)-1)}
            # Add reverse edges too (for undirected graph)
            path_edge_set.update({(last_path[i+1], last_path[i]) for i in range(len(last_path)-1)})
        
        # Define a fixed colormap for load visualization
        cmap = plt.cm.RdYlGn_r  # Red-Yellow-Green (reversed)
        
        # Style each edge
        for u, v, data in edge_list:
            # Extract edge attributes
            load = data.get('load', 0)
            capacity = data.get('capacity', 1)
            latency = data.get('latency', 0)
            bandwidth = data.get('bandwidth', 0)
            
            # Calculate load ratio for color
            load_ratio = min(1.0, load / capacity) if capacity > 0 else 0
            
            # Set style based on path and load
            if (u, v) in path_edge_set:
                # Edge is in active path
                color = '#ff9933'  # Orange
                width = 3.0
                style = 'solid'
                alpha = 1.0
            else:
                # Normal edge - color by load
                color = cmap(load_ratio)
                width = 1.0 + load_ratio  # Width increases with load
                style = 'solid'
                alpha = 0.7 + (0.3 * load_ratio)  # More opaque with more load
            
            edge_colors.append(color)
            edge_widths.append(width)
            edge_styles.append(style)
            edge_alphas.append(alpha)
        
        # Draw edges
        edges = nx.draw_networkx_edges(
            self.network, 
            self.network_pos,
            ax=self.ax_network,
            edge_color=edge_colors,
            width=edge_widths,
            style=edge_styles,
            alpha=edge_alphas
        )
        
        # 2. Draw nodes
        node_colors = []
        node_sizes = []
        node_shapes = []
        node_labels = {}
        node_alphas = []
        
        requesting_user_node = self.last_request_info.get('user')
        cached_at = self.last_request_info.get('cached_at')
        
        for node in self.network.nodes():
            node_type = self.network.nodes[node]['type']
            node_label = self.network.nodes[node].get('label', node)  # Use custom label if defined
            node_labels[node] = node_label
            
            if node_type == 'user':
                # User node styling
                shape = 'o'  # Circle
                if node == requesting_user_node:
                    # Active user
                    color = '#FFFF00'  # Bright Yellow
                    size = 800
                    alpha = 1.0
                else:
                    # Inactive user
                    color = '#ffcc66'  # Standard Yellow
                    size = 700
                    alpha = 0.8
            elif node_type == 'origin':
                # Origin server
                color = '#ff6666'  # Red
                size = 800
                shape = 's'  # Square
                alpha = 1.0
            elif node_type == 'edge':
                # Edge server styling
                shape = '^'  # Triangle
                if node == cached_at:
                    # The edge server is serving content
                    color = '#00cc00'  # Bright Green
                    size = 700
                    alpha = 1.0
                elif self.edge_server_flash[node] > 0:
                    # Flash when caching new content
                    color = '#66ff66'  # Light Green
                    size = 600
                    alpha = 1.0
                    self.edge_server_flash[node] -= 1
                else:
                    # Normal state
                    color = '#66b3ff'  # Blue
                    size = 500
                    alpha = 0.8
            else:
                # Default styling
                color = 'gray'
                size = 300
                shape = 'o'
                alpha = 0.7
            
            node_colors.append(color)
            node_sizes.append(size)
            node_shapes.append(shape)
            node_alphas.append(alpha)
        
        # Group nodes by shape for drawing
        shape_groups = {}
        for i, node in enumerate(self.network.nodes()):
            shape = node_shapes[i]
            if shape not in shape_groups:
                shape_groups[shape] = {'nodes': [], 'colors': [], 'sizes': [], 'alphas': []}
            shape_groups[shape]['nodes'].append(node)
            shape_groups[shape]['colors'].append(node_colors[i])
            shape_groups[shape]['sizes'].append(node_sizes[i])
            shape_groups[shape]['alphas'].append(node_alphas[i])
        
        # Draw each shape group separately
        for shape, data in shape_groups.items():
            nx.draw_networkx_nodes(
                self.network,
                self.network_pos,
                ax=self.ax_network,
                nodelist=data['nodes'],
                node_color=data['colors'],
                node_size=data['sizes'],
                node_shape=shape,
                alpha=data['alphas']
            )
        
        # 3. Draw labels
        nx.draw_networkx_labels(
            self.network,
            self.network_pos,
            ax=self.ax_network,
            labels=node_labels,
            font_size=8,
            font_weight='bold',
            font_color='black'
        )
        
        # 4. Add load legend
        # Create a custom legend for load levels
        legend_elements = [
            plt.Line2D([0], [0], color=cmap(0.0), lw=2, label='No Load'),
            plt.Line2D([0], [0], color=cmap(0.3), lw=2, label='Low Load'),
            plt.Line2D([0], [0], color=cmap(0.6), lw=2, label='Medium Load'),
            plt.Line2D([0], [0], color=cmap(0.9), lw=2, label='High Load'),
            plt.Line2D([0], [0], color='#ff9933', lw=3, label='Active Path')
        ]
        self.ax_network.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        # 5. Add annotations about the current request and content
        content_id = None
        if self.last_request_info:
            content_id = next((k for k, v in self.metrics.requests_per_content.items() 
                              if self.metrics.requests_per_content[k] == self.current_step), None)
        
        if content_id is not None:
            content_info = self.data_gen.get_content_info(content_id)
            score = content_info.get('current_score', 'N/A')
            size = content_info.get('size', 'N/A')
            
            # Add request details text box
            request_text = f"Request #{self.current_step}\nUser: {requesting_user_node}\nContent: {content_id}\nSize: {size}MB\nScore: {score:.1f}"
            self.ax_network.text(0.02, 0.98, request_text, transform=self.ax_network.transAxes,
                                fontsize=9, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Set title and remove axis ticks (not needed for network viz)
        delivery_time = self.last_request_info.get('delivery_time', None)
        if delivery_time:
            title = f'Network Topology (Step: {self.current_step}) - Delivery Time: {delivery_time:.2f}ms'
        else:
            title = f'Network Topology (Step: {self.current_step})'
            
        self.ax_network.set_title(title)
        self.ax_network.set_xticks([])
        self.ax_network.set_yticks([])

        # --- Update Engagement Score Plot (ax_engagement) ---
        self.ax_engagement.cla()
        scores = self.data_gen.get_all_current_scores()
        content_ids = list(range(self.num_content))
        colors = ['#ff6347' if s > self.viral_threshold else '#4682b4' for s in scores] # Tomato for viral, SteelBlue otherwise
        self.engagement_bars = self.ax_engagement.bar(content_ids, scores, color=colors)
        self.ax_engagement.axhline(y=self.viral_threshold, color='r', linestyle='--', label=f'Viral Threshold ({self.viral_threshold})')
        self.ax_engagement.set_ylim(0, 105)
        self.ax_engagement.set_title('Content Engagement Scores')
        self.ax_engagement.set_xlabel('Content ID')
        self.ax_engagement.set_ylabel('Engagement Score')
        self.ax_engagement.legend(loc='upper right')
        self.ax_engagement.grid(axis='y')

        # --- Update Delivery Time Plot (ax_delivery) ---
        times = list(self.delivery_time_history)
        if times:
            self.delivery_line.set_data(np.arange(len(times)), times)
            # Adjust y-axis limits dynamically
            min_time = np.min(times) if times else 0
            max_time = np.max(times) if times else 200
            self.ax_delivery.set_ylim(min_time * 0.9, max_time * 1.1 + 1) # Add padding (+1 for zero case)
            self.ax_delivery.set_xlim(0, self.delivery_time_history.maxlen)
            self.ax_delivery.figure.canvas.draw() # Needed for limit changes

        # --- Update Cache Performance Plot (ax_cache) ---
        self.ax_cache.cla() # Clear axes before drawing new bars
        metrics_summary = self.metrics.get_metrics_summary()
        hits = metrics_summary['cache_hits']
        misses = metrics_summary['cache_misses']
        self.cache_bars = self.ax_cache.bar([0, 1], [hits, misses], color=['#99ff99', '#ff9999'])
        self.ax_cache.set_xticks([0, 1])
        self.ax_cache.set_xticklabels(['Hits', 'Misses'])
        self.ax_cache.set_title(f'Cache Performance (Hit Rate: {metrics_summary["hit_rate"]:.1f}%)')
        self.ax_cache.set_ylabel('Count')
        # Add text labels on bars
        for bar in self.cache_bars:
            yval = bar.get_height()
            self.ax_cache.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05 * max(hits, misses, 1), int(yval), va='bottom', ha='center')
        self.ax_cache.set_ylim(0, max(hits, misses, 1) * 1.1) # Dynamic Y limit
        self.ax_cache.grid(axis='y')

        self.fig.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout slightly

        return () # Simplest approach if not using blitting


    def run_simulation(self, num_requests: int = 2000, interval_ms: int = 100):
       
        self.num_requests = num_requests
        print(f"Starting simulation for {num_requests} requests...")

        # Create the animation
        self.anim = animation.FuncAnimation(
            self.fig,
            self._update_plot,
            frames=num_requests + 10, # Add a few extra frames to ensure last step is shown
            interval=interval_ms,
            blit=False, # Blitting can be complex with networkx/clearing axes, set to False
            repeat=False
            )

        plt.show() # Display the animation window

        # --- Post-Simulation Analysis ---
        print("\nSimulation finished.")
        self.metrics.display_final_metrics() # Show summary stats and plots
        self._display_content_summary() # Show detailed content info table


    def _display_content_summary(self):
         """Displays a summary table of content performance."""
         data = []
         all_scores = self.data_gen.get_all_current_scores()
         for content_id in range(self.num_content):
             info = self.data_gen.get_content_info(content_id)
             total_reqs = self.metrics.requests_per_content.get(content_id, 0)
             edge_reqs = self.metrics.edge_hits_per_content.get(content_id, 0)
             # Note: Avg delivery time per content isn't directly tracked; using overall avg
             avg_delivery_time = self.metrics.get_metrics_summary()['avg_delivery_time']

             data.append({
                 'Content ID': content_id,
                 'Final Score': f"{all_scores[content_id]:.1f}",
                 'Size (MB)': info.get('size', 'N/A'),
                 'Total Requests': total_reqs,
                 'Served by Edge': edge_reqs, # Hits served by edge cache
                 'Viral': 'Yes' if all_scores[content_id] > self.viral_threshold else 'No'
                 # 'Avg Delivery Time': f"{avg_delivery_time:.2f} ms" # Maybe misleading per content
             })

         df = pd.DataFrame(data)
         # Sort by total requests descending for relevance
         df = df.sort_values(by='Total Requests', ascending=False)

         print("\n--- Content Information Summary ---")
         if not df.empty:
              print(df.to_string(index=False))
         else:
              print("No content data to display.")


# --- Main Execution ---
if __name__ == "__main__":
    cdn_simulation = AdaptiveCDN(
        num_servers=8,
        num_users=5,         # Specify number of users
        cache_capacity=50,
        viral_threshold=75,
        data_file='initial_content_data.json',
        load_decay_factor=0.1, # Example decay factor (10% per step)
        congestion_penalty=50  # Example congestion penalty for router
    )
    cdn_simulation.run_simulation(
        num_requests=1000,   # Increased requests to see effects
        interval_ms=150      # Slightly faster animation
    )