from __future__ import annotations

import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import networkx as nx


class ConsistencyMode:
    """Enumeration for consistency modes"""
    WEAK = 0
    STRONG = 1


class EvaluationMetrics:
    """Container for evaluation metrics for a single strategy"""

    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name
        self.latencies: List[float] = []
        self.slo_violations: int = 0
        self.total_consumptions: int = 0
        self.strong_consistency_latencies: List[float] = []
        self.weak_consistency_latencies: List[float] = []
        self.strong_consistency_violations: int = 0
        self.weak_consistency_violations: int = 0

    def add_latency(
            self,
            latency: float,
            slo: float,
            consistency_mode: Optional[int] = None
    ):
        """Record a latency measurement"""
        self.latencies.append(latency)
        self.total_consumptions += 1

        if latency > slo:
            self.slo_violations += 1

        # Track consistency-specific metrics if provided
        if consistency_mode == ConsistencyMode.STRONG:
            self.strong_consistency_latencies.append(latency)
            if latency > slo:
                self.strong_consistency_violations += 1
        elif consistency_mode == ConsistencyMode.WEAK:
            self.weak_consistency_latencies.append(latency)
            if latency > slo:
                self.weak_consistency_violations += 1

    def get_summary(self) -> Dict:
        """Get summary statistics"""
        return {
            "average_latency": np.mean(self.latencies) if self.latencies else float('inf'),
            "median_latency": np.median(self.latencies) if self.latencies else float('inf'),
            "p95_latency": np.percentile(self.latencies, 95) if self.latencies else float('inf'),
            "slo_violation_rate": self.slo_violations / max(1, self.total_consumptions),
            "total_consumptions": self.total_consumptions
        }

    def get_consistency_breakdown(self) -> Dict:
        """Get consistency-specific breakdown"""
        return {
            "strong_consistency": {
                "average_latency": np.mean(self.strong_consistency_latencies)
                if self.strong_consistency_latencies else float('inf'),
                "median_latency": np.median(self.strong_consistency_latencies)
                if self.strong_consistency_latencies else float('inf'),
                "p95_latency": np.percentile(self.strong_consistency_latencies, 95)
                if self.strong_consistency_latencies else float('inf'),
                "slo_violation_rate": self.strong_consistency_violations /
                                      max(1, len(self.strong_consistency_latencies)),
                "total_consumptions": len(self.strong_consistency_latencies)
            },
            "weak_consistency": {
                "average_latency": np.mean(self.weak_consistency_latencies)
                if self.weak_consistency_latencies else float('inf'),
                "median_latency": np.median(self.weak_consistency_latencies)
                if self.weak_consistency_latencies else float('inf'),
                "p95_latency": np.percentile(self.weak_consistency_latencies, 95)
                if self.weak_consistency_latencies else float('inf'),
                "slo_violation_rate": self.weak_consistency_violations /
                                      max(1, len(self.weak_consistency_latencies)),
                "total_consumptions": len(self.weak_consistency_latencies)
            }
        }


class EvaluationManager:
    """
    Manages evaluation of placement strategies and consumption patterns
    Tracks metrics, performs consumption testing, and generates reports
    """

    def __init__(
            self,
            topology: nx.Graph,
            enable_consistency: bool = True,
            enable_congestion: bool = True
    ):
        self.topology = topology
        self.enable_consistency = enable_consistency
        self.enable_congestion = enable_congestion

        # Strategy-specific metrics
        self.strategy_metrics: Dict[str, EvaluationMetrics] = {}

        # Global metrics
        self.data_generated = 0
        self.data_consumed = 0
        self.total_bytes_transferred = 0
        self.storage_usage_history: List[float] = []
        self.congestion_history: List[float] = []

    def register_strategy(self, strategy_name: str):
        """Register a new strategy for tracking"""
        if strategy_name not in self.strategy_metrics:
            self.strategy_metrics[strategy_name] = EvaluationMetrics(strategy_name)

    def record_data_generation(self):
        """Record that data was generated"""
        self.data_generated += 1

    def record_data_consumption(self):
        """Record that data was consumed"""
        self.data_consumed += 1

    def test_consumption_weak(
            self,
            strategy_name: str,
            consumer_node: int,
            data_item: Dict,
            consumer_instance: Dict,
            replica_key: str,
            latency_calculator,
            transfer_tracker
    ) -> Optional[float]:
        """
        Test consumption with weak consistency (read from closest replica)

        Returns:
            Latency if successful, None otherwise
        """
        replicas = data_item.get(replica_key, [])
        if not replicas:
            return None

        min_latency = float('inf')
        best_path = None
        best_replica = None

        # Find closest replica
        for replica_node in replicas:
            latency = latency_calculator(replica_node, consumer_node, data_item["size_bytes"])
            if latency < min_latency and np.isfinite(latency):
                min_latency = latency
                best_replica = replica_node

        if min_latency < float('inf') and best_replica:
            # Track data transfer
            transfer_tracker(best_replica, consumer_node, data_item["size_bytes"])
            self.total_bytes_transferred += data_item["size_bytes"]

            # Record metrics
            consistency_mode = ConsistencyMode.WEAK if self.enable_consistency else None
            self.strategy_metrics[strategy_name].add_latency(
                min_latency,
                consumer_instance.get("slo", float('inf')),
                consistency_mode
            )

            return min_latency

        return None

    def test_consumption_strong(
            self,
            strategy_name: str,
            consumer_node: int,
            data_item: Dict,
            consumer_instance: Dict,
            replica_key: str,
            latency_calculator,
            transfer_tracker
    ) -> Optional[float]:
        """
        Test consumption with strong consistency (read from ALL replicas)

        Returns:
            Maximum latency if successful, None otherwise
        """
        replicas = data_item.get(replica_key, [])
        if not replicas:
            return None

        max_latency = 0
        total_bytes_transferred = 0

        # Read from ALL replicas
        for replica_node in replicas:
            latency = latency_calculator(replica_node, consumer_node, data_item["size_bytes"])

            if not np.isfinite(latency):
                continue

            # Track data transfer (each replica transfer counts)
            transfer_tracker(replica_node, consumer_node, data_item["size_bytes"])
            total_bytes_transferred += data_item["size_bytes"]

            # For strong consistency, effective latency is the maximum
            max_latency = max(max_latency, latency)

        if max_latency > 0 and np.isfinite(max_latency):
            self.total_bytes_transferred += total_bytes_transferred

            # Record metrics
            consistency_mode = ConsistencyMode.STRONG if self.enable_consistency else None
            self.strategy_metrics[strategy_name].add_latency(
                max_latency,
                consumer_instance.get("slo", float('inf')),
                consistency_mode
            )

            return max_latency

        return None

    def test_consumption_single_replica(
            self,
            strategy_name: str,
            consumer_node: int,
            data_item: Dict,
            consumer_instance: Dict,
            replica_key: str,
            latency_calculator,
            transfer_tracker
    ) -> Optional[float]:
        """
        Test consumption for single-replica strategies (e.g., local placement)

        Returns:
            Latency if successful, None otherwise
        """
        replicas = data_item.get(replica_key, [])
        if not replicas:
            return None

        # Use the first (only) replica
        replica_node = replicas[0]

        latency = latency_calculator(replica_node, consumer_node, data_item["size_bytes"])

        if np.isfinite(latency):
            # Track data transfer
            transfer_tracker(replica_node, consumer_node, data_item["size_bytes"])
            self.total_bytes_transferred += data_item["size_bytes"]

            # Record metrics
            self.strategy_metrics[strategy_name].add_latency(
                latency,
                consumer_instance.get("slo", float('inf')),
                consistency_mode=None
            )

            return latency

        return None

    def record_storage_usage(self, usage_percentage: float):
        """Record storage usage snapshot"""
        self.storage_usage_history.append(usage_percentage)

    def record_congestion(self, avg_utilization: float):
        """Record network congestion snapshot"""
        self.congestion_history.append(avg_utilization)

    def get_results(self) -> Dict:
        """Get comprehensive results for all strategies"""
        results = {
            "strategies": {},
            "data_throughput": {
                "generated": self.data_generated,
                "consumed": self.data_consumed
            },
            "storage": {
                "average_usage_percentage": np.mean(self.storage_usage_history)
                if self.storage_usage_history else 0,
                "final_usage_percentage": self.storage_usage_history[-1]
                if self.storage_usage_history else 0
            },
            "network": {
                "average_congestion": np.mean(self.congestion_history)
                if self.congestion_history else 0,
                "max_congestion": np.max(self.congestion_history)
                if self.congestion_history else 0,
                "total_bytes_transferred": self.total_bytes_transferred
            },
            "consistency_enabled": self.enable_consistency,
            "congestion_enabled": self.enable_congestion
        }

        # Add strategy-specific results
        for strategy_name, metrics in self.strategy_metrics.items():
            results["strategies"][strategy_name] = metrics.get_summary()

            # Add consistency breakdown if enabled
            if self.enable_consistency and (
                    metrics.strong_consistency_latencies or metrics.weak_consistency_latencies
            ):
                results["strategies"][strategy_name]["consistency_breakdown"] = \
                    metrics.get_consistency_breakdown()

        return results

    def print_results(self, results: Optional[Dict] = None):
        """Print formatted results"""
        if results is None:
            results = self.get_results()

        print("\n=== EVALUATION RESULTS ===")
        print(f"Data generated: {results['data_throughput']['generated']}")
        print(f"Data consumed: {results['data_throughput']['consumed']}")
        print(f"Storage usage: {results['storage']['average_usage_percentage']:.2f}% (avg)")
        print(f"Network - Avg Congestion: {results['network']['average_congestion']:.3f}")
        print(f"Total bytes transferred: {results['network']['total_bytes_transferred']:,.0f} bytes")

        # Print each strategy's results
        for strategy_name, strategy_results in results["strategies"].items():
            print(f"\n--- Strategy: {strategy_name} ---")
            print(f"Average latency: {strategy_results['average_latency']:.2f} ms")
            print(f"Median latency: {strategy_results['median_latency']:.2f} ms")
            print(f"P95 latency: {strategy_results['p95_latency']:.2f} ms")
            print(f"SLO violation rate: {strategy_results['slo_violation_rate']:.2%}")
            print(f"Total consumptions: {strategy_results['total_consumptions']}")

            # Print consistency breakdown if available
            if "consistency_breakdown" in strategy_results:
                breakdown = strategy_results["consistency_breakdown"]
                print("\n  Consistency Breakdown:")

                strong = breakdown["strong_consistency"]
                print(f"  STRONG (read all replicas):")
                print(f"    Avg latency: {strong['average_latency']:.2f} ms")
                print(f"    SLO violations: {strong['slo_violation_rate']:.2%}")
                print(f"    Consumptions: {strong['total_consumptions']}")

                weak = breakdown["weak_consistency"]
                print(f"  WEAK (read closest):")
                print(f"    Avg latency: {weak['average_latency']:.2f} ms")
                print(f"    SLO violations: {weak['slo_violation_rate']:.2%}")
                print(f"    Consumptions: {weak['total_consumptions']}")

    def compare_strategies(self, results: Optional[Dict] = None) -> Dict:
        """Compare strategies and identify best performing"""
        if results is None:
            results = self.get_results()

        strategies = results["strategies"]
        if not strategies:
            return {}

        # Find best strategy by average latency
        best_strategy = None
        best_latency = float('inf')

        for strategy_name, strategy_results in strategies.items():
            avg_latency = strategy_results['average_latency']
            if np.isfinite(avg_latency) and avg_latency < best_latency:
                best_latency = avg_latency
                best_strategy = strategy_name

        comparison = {
            "best_strategy": best_strategy,
            "best_latency": best_latency,
            "relative_performance": {}
        }

        # Calculate relative performance
        for strategy_name, strategy_results in strategies.items():
            if strategy_name == best_strategy:
                comparison["relative_performance"][strategy_name] = 0.0
            else:
                avg_latency = strategy_results['average_latency']
                if np.isfinite(avg_latency) and best_latency > 0:
                    diff_pct = ((avg_latency - best_latency) / avg_latency) * 100
                    comparison["relative_performance"][strategy_name] = diff_pct

        return comparison

    def reset(self):
        """Reset all metrics"""
        self.strategy_metrics.clear()
        self.data_generated = 0
        self.data_consumed = 0
        self.total_bytes_transferred = 0
        self.storage_usage_history.clear()
        self.congestion_history.clear()