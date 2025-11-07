from __future__ import annotations

import itertools
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import networkx as nx
    from topology import ShortestPathCache


class PlacementContext:
    """Context object containing all data needed for placement decisions"""

    def __init__(
            self,
            topology: nx.Graph,
            instance_manager,
            latency_cache: ShortestPathCache,
            storage_usage: Dict[int, Dict],
            rng: random.Random
    ):
        self.topology = topology
        self.instance_manager = instance_manager
        self.latency_cache = latency_cache
        self.storage_usage = storage_usage
        self.rng = rng

    def has_storage_space(self, node_id: int, data_size_gb: float) -> bool:
        """Check if node has enough storage space"""
        return self.storage_usage[node_id]["free"] >= data_size_gb

    def get_available_storage_nodes(self, min_space_gb: float) -> List[int]:
        """Get all storage nodes with at least min_space_gb available"""
        return [
            node_id for node_id, usage in self.storage_usage.items()
            if self.has_storage_space(node_id, min_space_gb)
        ]


class PlacementStrategy(ABC):
    """Abstract base class for data placement strategies"""

    @abstractmethod
    def place(
            self,
            data_id: str,
            data_item: Dict,
            context: PlacementContext
    ) -> List[int]:
        """
        Place data and return list of storage node IDs where replicas are placed

        Args:
            data_id: Unique identifier for the data item
            data_item: Dictionary containing data metadata (size, generator info, etc.)
            context: PlacementContext with topology and storage info

        Returns:
            List of node IDs where replicas were placed
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return strategy name for logging/identification"""
        pass


class LocalPlacementStrategy(PlacementStrategy):
    """
    Approach 1: Store data locally or on nearest storage node
    Single replica, minimal latency for writes
    """

    def get_name(self) -> str:
        return "local_storage"

    def place(
            self,
            data_id: str,
            data_item: Dict,
            context: PlacementContext
    ) -> List[int]:
        generator_node = data_item["generator_node"]
        data_size_gb = data_item["size_gb"]

        # Check if generator node is a storage node
        node_data = context.topology.nodes[generator_node]
        if node_data.get("type") in ["cloud", "fog", "edge"]:
            target_node = generator_node
        else:
            target_node = self._find_nearest_storage_node(generator_node, context)

        # Place on target node if it has space
        if target_node and context.has_storage_space(target_node, data_size_gb):
            return [target_node]

        return []

    def _find_nearest_storage_node(
            self,
            start_node: int,
            context: PlacementContext
    ) -> Optional[int]:
        """Find the nearest storage node to the given node"""
        storage_nodes = [
            n for n, d in context.topology.nodes(data=True)
            if d.get("type") in ["cloud", "fog", "edge"]
        ]

        if not storage_nodes:
            return None

        min_latency = float('inf')
        closest_node = None

        for storage_node in storage_nodes:
            latency = context.latency_cache.get_latency(start_node, storage_node)
            if latency < min_latency:
                min_latency = latency
                closest_node = storage_node

        return closest_node


class RandomReplicaPlacementStrategy(PlacementStrategy):
    """
    Approach 2: Store 3 replicas on random storage nodes
    Provides redundancy and geographic distribution
    """

    def __init__(self, num_replicas: int = 3):
        self.num_replicas = num_replicas

    def get_name(self) -> str:
        return f"random_{self.num_replicas}_replicas"

    def place(
            self,
            data_id: str,
            data_item: Dict,
            context: PlacementContext
    ) -> List[int]:
        data_size_gb = data_item["size_gb"]

        # Get all storage nodes with enough space
        available_nodes = context.get_available_storage_nodes(data_size_gb)

        # Select random nodes (or as many as available)
        num_replicas = min(self.num_replicas, len(available_nodes))
        if num_replicas > 0:
            return context.rng.sample(available_nodes, num_replicas)

        return []


class PCenterPlacementStrategy(PlacementStrategy):
    """
    Approach 3: p-Center placement - minimize maximum latency to consumers
    Places replicas to optimize read latency for known consumers
    """

    def __init__(self, num_replicas: int = 3):
        self.num_replicas = num_replicas

    def get_name(self) -> str:
        return f"p_center_{self.num_replicas}_replicas"

    def place(
            self,
            data_id: str,
            data_item: Dict,
            context: PlacementContext
    ) -> List[int]:
        generator_instance_id = data_item["generator_instance"]
        data_size_gb = data_item["size_gb"]

        generator_instance = context.instance_manager.get_instance(generator_instance_id)
        connected_consumers = generator_instance["connected_consumers"]

        if not connected_consumers:
            # Fall back to random if no connected consumers
            fallback = RandomReplicaPlacementStrategy(self.num_replicas)
            return fallback.place(data_id, data_item, context)

        # Get consumer nodes
        consumer_nodes = []
        for consumer_instance_id in connected_consumers:
            consumer_instance = context.instance_manager.get_instance(consumer_instance_id)
            consumer_nodes.append(consumer_instance["node_id"])

        # Get available storage nodes
        available_nodes = context.get_available_storage_nodes(data_size_gb)

        if len(available_nodes) < self.num_replicas:
            return available_nodes  # Return all available if less than p

        # Find p nodes that minimize the maximum latency to any consumer
        best_replicas = self._find_optimal_placement(
            available_nodes,
            consumer_nodes,
            context.latency_cache
        )

        return best_replicas if best_replicas else []

    def _find_optimal_placement(
            self,
            available_nodes: List[int],
            consumer_nodes: List[int],
            latency_cache: ShortestPathCache
    ) -> List[int]:
        """
        Find optimal p-center placement using brute force for small p
        For larger p, could use approximation algorithms
        """
        best_replicas = None
        best_max_latency = float('inf')

        # Try all combinations (feasible for small p like 3)
        for replica_set in itertools.combinations(available_nodes, self.num_replicas):
            max_latency = 0

            for consumer_node in consumer_nodes:
                # For each consumer, find minimum latency to any replica
                consumer_min_latency = min(
                    latency_cache.get_latency(consumer_node, replica)
                    for replica in replica_set
                )
                # Track the maximum of these minimums
                max_latency = max(max_latency, consumer_min_latency)

            if max_latency < best_max_latency:
                best_max_latency = max_latency
                best_replicas = replica_set

        return list(best_replicas) if best_replicas else []


# Factory function for easy strategy creation
def create_placement_strategy(strategy_type: str, **kwargs) -> PlacementStrategy:
    """
    Factory function to create placement strategies

    Args:
        strategy_type: One of 'local', 'random', 'p_center', 'latency_aware', 'hierarchical'
        **kwargs: Additional parameters (e.g., num_replicas=3)

    Returns:
        PlacementStrategy instance
    """
    strategies = {
        'local': LocalPlacementStrategy,
        'random': RandomReplicaPlacementStrategy,
        'p_center': PCenterPlacementStrategy,
        'latency_aware': LatencyAwarePlacementStrategy,
        'hierarchical': HierarchicalPlacementStrategy
    }

    if strategy_type not in strategies:
        raise ValueError(f"Unknown strategy type: {strategy_type}. "
                         f"Available: {list(strategies.keys())}")

    strategy_class = strategies[strategy_type]

    # Local strategy doesn't take num_replicas
    if strategy_type == 'local':
        return strategy_class()
    elif strategy_type == 'hierarchical':
        return strategy_class()
    else:
        num_replicas = kwargs.get('num_replicas', 3)
        return strategy_class(num_replicas=num_replicas)