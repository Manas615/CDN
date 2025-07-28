from CDN import setup_environment
from data_generation import DataGenerator
from Routing_algorithm import EngagementAwareRouter
from caching_stratergy import ViralLFUCache
from performance_matrix import PerformanceMetrics
def run_simulation(num_requests=1000, cache_capacity=20):
    # Setup environment
    env, network = setup_environment()

    # Initialize components
    data_gen = DataGenerator()
    router = EngagementAwareRouter(network)
    cache = ViralLFUCache(capacity=cache_capacity)
    metrics = PerformanceMetrics(len(network.nodes))

    for _ in range(num_requests):
        # Generate request
        content_id, engagement_score = data_gen.generate_request()

        # Route content
        path = router.route_content('User', content_id)

        # Check cache and update metrics
        if cache.get(content_id) is not None:
            metrics.record_cache_hit()
        else:
            metrics.record_cache_miss()
            cache.put(content_id, f"Content_{content_id}", engagement_score)

        # Update metrics
        delivery_time = data_gen.simulate_delivery_time(path)
        metrics.record_delivery_time(delivery_time)
        metrics.record_engagement_score(engagement_score)
        metrics.record_request_cost(path)

        # Simulate network conditions
        if data_gen.simulate_network_congestion():
            metrics.record_timeout()

        # Update components periodically
        if _ % 100 == 0:
            data_gen.update_network_conditions()
            router.update_network_conditions(data_gen.get_network_conditions())
            cache.update_viral_status()

    return metrics

# Run the simulation
if __name__ == "__main__":
    final_metrics = run_simulation()
    print(final_metrics.get_metrics_summary())
