import time
from collections import defaultdict

class ViralLFUCache:
    def __init__(self, capacity, viral_threshold=100, time_window=3600):
        self.capacity = capacity
        self.viral_threshold = viral_threshold
        self.time_window = time_window  # in seconds
        self.cache = {}
        self.frequency = defaultdict(int)
        self.last_access = {}
        self.viral_content = set()

    def get(self, key):
        if key in self.cache:
            self.frequency[key] += 1
            self.last_access[key] = time.time()
            return self.cache[key]
        return None

    def put(self, key, value, engagement_score):
        current_time = time.time()

        # Check if content is potentially viral
        if engagement_score > self.viral_threshold:
            self.viral_content.add(key)

        if key in self.cache:
            self.cache[key] = value
            self.frequency[key] += 1
        else:
            if len(self.cache) >= self.capacity:
                self.evict()
            self.cache[key] = value
            self.frequency[key] = 1

        self.last_access[key] = current_time

    def evict(self):
        current_time = time.time()
        min_freq = min(self.frequency.values())
        lfu_keys = [k for k, v in self.frequency.items() if v == min_freq]

        # Sort LFU keys by last access time, oldest first
        lfu_keys.sort(key=lambda x: self.last_access[x])

        for key in lfu_keys:
            if key not in self.viral_content:
                del self.cache[key]
                del self.frequency[key]
                del self.last_access[key]
                return

        # If all LFU content is viral, remove the oldest one
        oldest_key = min(self.last_access, key=self.last_access.get)
        del self.cache[oldest_key]
        del self.frequency[oldest_key]
        del self.last_access[oldest_key]
        self.viral_content.discard(oldest_key)

    def update_viral_status(self):
        current_time = time.time()
        for key in list(self.viral_content):
            if current_time - self.last_access[key] > self.time_window:
                self.viral_content.discard(key)

def simulate_cache_behavior(cache, num_requests=1000):
    import random

    content_pool = list(range(1, 101))  # Content IDs from 1 to 100
    viral_content = random.sample(content_pool, 5)  # 5 random viral content

    cache_hits = 0
    cache_misses = 0

    for _ in range(num_requests):
        if random.random() < 0.2:  # 20% chance of requesting viral content
            content_id = random.choice(viral_content)
            engagement_score = random.randint(100, 200)
        else:
            content_id = random.choice(content_pool)
            engagement_score = random.randint(1, 99)

        if cache.get(content_id) is not None:
            cache_hits += 1
        else:
            cache_misses += 1
            cache.put(content_id, f"Content_{content_id}", engagement_score)

        if _ % 100 == 0:
            cache.update_viral_status()

    return cache_hits, cache_misses

# Main execution
if __name__ == "__main__":
    cache = ViralLFUCache(capacity=20, viral_threshold=100, time_window=3600)
    hits, misses = simulate_cache_behavior(cache)

    print(f"Cache Hits: {hits}")
    print(f"Cache Misses: {misses}")
    print(f"Hit Rate: {hits / (hits + misses):.2%}")
    print(f"Number of viral content in cache: {len(cache.viral_content)}")
