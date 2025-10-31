from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import networkx as nx
import numpy as np
from topology import ShortestPathCache
from placement_strategies import PlacementStrategy, PlacementContext
from evaluation import EvaluationManager, ConsistencyMode


class DataContinuumEnv:
    """
    Isolated environment for a SINGLE placement strategy
    Each strategy runs in its own independent environment instance
    """

    def __init__(
            self,
            topology: nx.Graph,
            instance_manager,
            placement_strategy: PlacementStrategy,
            *,
            enable_congestion: bool = True,
            enable_consistency: bool = True,
            time_step_duration: float = 10,
            random_state: Optional[int] = 42,
            max_steps: int = 1000
    ):
        """
        Args:
            topology: NetworkX graph of the infrastructure
            instance_manager: Application instance manager
            placement_strategy: Single PlacementStrategy instance for this environment
            enable_congestion: Whether to model network congestion
            enable_consistency: Whether to model strong/weak consistency
            time_step_duration: Duration of each time step in seconds
            random_state: Random seed
            max_steps: Maximum simulation steps
        """
        self.topology = topology
        self.instance_manager = instance_manager
        self.placement_strategy = placement_strategy
        self.strategy_name = placement_strategy.get_name()
        self.enable_congestion = enable_congestion
        self.enable_consistency = enable_consistency
        self.time_step_duration = time_step_duration
        self.random_state = random_state
        self.max_steps = max_steps

        # Initialize RNG
        self.rng = random.Random(random_state)

        # Track data items (specific to this strategy)
        self.data_catalogue: Dict[str, Dict] = {}
        self.next_data_id = 1

        # Track storage usage (copy for isolation)
        self._initialize_storage_usage()

        # Time tracking
        self.current_step = 0

        # Create latency cache
        self.latency_cache = ShortestPathCache(topology)

        # Create placement context (isolated for this strategy)
        self.placement_context = PlacementContext(
            topology=topology,
            instance_manager=instance_manager,
            latency_cache=self.latency_cache,
            storage_usage=self.storage_usage,
            rng=self.rng
        )

        # Track generation and consumption timers
        self.generator_timers: Dict[int, int] = {}
        self.consumer_timers: Dict[int, int] = {}

        # Congestion tracking
        self.link_utilization: Dict[Tuple[int, int], float] = defaultdict(float)
        self.current_latencies: Dict[Tuple[int, int], float] = {}
        self._initialize_latencies()

        # Initialize timers
        self._initialize_timers()

        # Initialize evaluation manager (one per environment)
        self.evaluation = EvaluationManager(
            topology=topology,
            enable_consistency=enable_consistency,
            enable_congestion=enable_congestion
        )

        # Register the single strategy
        self.evaluation.register_strategy(self.strategy_name)

    def _initialize_storage_usage(self):
        """Initialize storage tracking for all storage nodes"""
        self.storage_usage: Dict[int, Dict] = {}

        for node_id, node_data in self.topology.nodes(data=True):
            if node_data.get("type") in ["cloud", "fog", "edge"]:
                self.storage_usage[node_id] = {
                    "used": 0,
                    "free": node_data.get("storage_capacity", 0),
                    "stored_data": set()
                }

    def _initialize_latencies(self):
        """Initialize latencies with base values from topology"""
        for u, v, data in self.topology.edges(data=True):
            self.current_latencies[(u, v)] = data.get("latency", 1.0)
            self.current_latencies[(v, u)] = data.get("latency", 1.0)

    def _initialize_timers(self):
        """Initialize generation and consumption timers with random offsets"""
        for instance_id in self.instance_manager.generator_instances:
            instance = self.instance_manager.get_instance(instance_id)
            self.generator_timers[instance_id] = self.rng.randint(0, instance["generation_rate"] - 1)

        for instance_id in self.instance_manager.consumer_instances:
            instance = self.instance_manager.get_instance(instance_id)
            self.consumer_timers[instance_id] = self.rng.randint(0, instance["consumption_rate"] - 1)

    def reset(self):
        """Reset the environment to initial state"""
        self.current_step = 0
        self.data_catalogue.clear()
        self.next_data_id = 1

        # Reset storage usage
        self._initialize_storage_usage()

        # Reset timers (with same seed for reproducibility)
        self.rng = random.Random(self.random_state)
        self._initialize_timers()

        # Reset congestion tracking
        self.link_utilization.clear()
        self._initialize_latencies()

        # Reset evaluation
        self.evaluation.reset()
        self.evaluation.register_strategy(self.strategy_name)

    def step(self):
        """Advance simulation by one timestep"""
        self.current_step += 1

        # Reset link utilization for new timestep
        if self.enable_congestion:
            self.link_utilization.clear()

        # 1. Generate new data from generator instances
        self._generate_data()

        # 2. Process consumption from consumer instances
        self._process_consumption()

        # 3. Update dynamic latencies based on current utilization
        if self.enable_congestion:
            self._update_dynamic_latencies()

        # 4. Track storage usage
        self._track_storage_usage()

        # 5. Track congestion levels
        self._track_congestion_levels()

        # 6. Check if simulation should end
        done = self.current_step >= self.max_steps

        return done

    def _generate_data(self):
        """Generate data from generator instances using the single placement strategy"""
        for instance_id in self.instance_manager.generator_instances:
            instance = self.instance_manager.get_instance(instance_id)

            # Decrement timer
            self.generator_timers[instance_id] -= 1

            # Check if it's time to generate
            if self.generator_timers[instance_id] <= 0:
                # Reset timer
                self.generator_timers[instance_id] = instance["generation_rate"]

                # Generate data
                data_id = f"data_{self.next_data_id}"
                self.next_data_id += 1

                data_item = {
                    "id": data_id,
                    "generator_instance": instance_id,
                    "generator_node": instance["node_id"],
                    "size_gb": instance["generated_size"],
                    "size_bytes": instance["generated_size"] * 1024 * 1024 * 1024,
                    "generation_time": self.current_step,
                    "replicas": []  # Single list for this strategy
                }

                # Store in catalogue
                self.data_catalogue[data_id] = data_item
                self.evaluation.record_data_generation()

                # Apply the single placement strategy
                selected_nodes = self.placement_strategy.place(
                    data_id, data_item, self.placement_context
                )

                # Store data on selected nodes
                for node_id in selected_nodes:
                    if node_id != instance["node_id"]:
                        # Track data transfer for congestion
                        path = self._get_shortest_path(instance["node_id"], node_id)
                        self._add_data_transfer(path, data_item["size_bytes"])

                    self._store_data_on_node(data_id, node_id, data_item["size_gb"])
                    data_item["replicas"].append(node_id)

    def _process_consumption(self):
        """Process data consumption from consumer instances"""
        for instance_id in self.instance_manager.consumer_instances:
            instance = self.instance_manager.get_instance(instance_id)

            # Decrement timer
            self.consumer_timers[instance_id] -= 1

            # Check if it's time to consume
            if self.consumer_timers[instance_id] <= 0:
                # Reset timer
                self.consumer_timers[instance_id] = instance["consumption_rate"]

                consumer_node = instance["node_id"]

                # For each connected generator, try to consume latest data
                for generator_instance_id in instance["connected_generators"]:
                    latest_data = self._find_latest_data_from_generator(generator_instance_id)

                    if latest_data:
                        self.evaluation.record_data_consumption()
                        self._test_consumption(consumer_node, latest_data, instance)

    def _test_consumption(
            self,
            consumer_node: int,
            data_item: Dict,
            consumer_instance: Dict
    ):
        """Test consumption for this strategy"""
        replicas = data_item.get("replicas", [])
        if not replicas:
            return

        # Determine consistency mode
        consistency_required = (
                self.enable_consistency and
                consumer_instance["consistency"] == ConsistencyMode.STRONG
        )

        # Single replica strategies always use single replica logic
        if len(replicas) == 1:
            self.evaluation.test_consumption_single_replica(
                strategy_name=self.strategy_name,
                consumer_node=consumer_node,
                data_item=data_item,
                consumer_instance=consumer_instance,
                replica_key="replicas",
                latency_calculator=self._calculate_latency,
                transfer_tracker=self._track_transfer
            )
        elif consistency_required:
            # Strong consistency: read from all replicas
            self.evaluation.test_consumption_strong(
                strategy_name=self.strategy_name,
                consumer_node=consumer_node,
                data_item=data_item,
                consumer_instance=consumer_instance,
                replica_key="replicas",
                latency_calculator=self._calculate_latency,
                transfer_tracker=self._track_transfer
            )
        else:
            # Weak consistency: read from closest replica
            self.evaluation.test_consumption_weak(
                strategy_name=self.strategy_name,
                consumer_node=consumer_node,
                data_item=data_item,
                consumer_instance=consumer_instance,
                replica_key="replicas",
                latency_calculator=self._calculate_latency,
                transfer_tracker=self._track_transfer
            )

    def _track_transfer(self, source_node: int, target_node: int, data_size_bytes: float):
        """Track a data transfer for congestion modeling"""
        path = self._get_shortest_path(source_node, target_node)
        if path:
            self._add_data_transfer(path, data_size_bytes)

    def _store_data_on_node(self, data_id: str, node_id: int, data_size: float):
        """Store data on specified node"""
        usage = self.storage_usage[node_id]
        usage["used"] += data_size
        usage["free"] -= data_size
        usage["stored_data"].add(data_id)

    def _find_latest_data_from_generator(self, generator_instance_id: int) -> Optional[Dict]:
        """Find the most recent data item from a generator instance"""
        latest_data = None
        latest_time = -1

        for data_id, data_item in self.data_catalogue.items():
            if data_item["generator_instance"] == generator_instance_id:
                if data_item["generation_time"] > latest_time:
                    latest_time = data_item["generation_time"]
                    latest_data = data_item

        return latest_data

    def _calculate_latency(self, source: int, target: int, data_size_bytes: float) -> float:
        """Calculate latency with congestion modeling"""
        if source == target:
            return 0.0

        path = self._get_shortest_path(source, target)
        if not path or len(path) < 2:
            return float('inf')

        total_latency = 0.0

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]

            link_data = self.topology[u][v]
            base_latency = link_data.get("latency", 1.0)
            bandwidth_bps = link_data.get("bandwidth", 1.0) * 1e6
            congestion_sensitivity = link_data.get("congestion_sensitivity", 1.0)

            # Transmission delay
            transmission_delay = (data_size_bytes * 8) / bandwidth_bps
            transmission_delay_ms = transmission_delay * 1000

            # Congestion delay
            congestion_delay = 0.0
            if self.enable_congestion:
                utilization = self.link_utilization.get((u, v), 0.0)
                if utilization < 1.0:
                    congestion_delay = congestion_sensitivity * (utilization / (1 - utilization))

            link_latency = base_latency + transmission_delay_ms + congestion_delay
            total_latency += link_latency

        return total_latency

    def _add_data_transfer(self, path: List[int], data_size_bytes: float):
        """Add data transfer to link utilization tracking"""
        if not path or len(path) < 2 or not self.enable_congestion:
            return

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]

            link_data = self.topology[u][v]
            bandwidth_bps = link_data.get("bandwidth", 1.0) * 1e6

            bytes_per_second = data_size_bytes / self.time_step_duration
            bits_per_second = bytes_per_second * 8
            utilization = bits_per_second / bandwidth_bps

            self.link_utilization[(u, v)] += utilization

    def _update_dynamic_latencies(self):
        """Update dynamic latencies based on current link utilization"""
        for (u, v), utilization in self.link_utilization.items():
            if (u, v) not in self.topology.edges:
                continue

            link_data = self.topology[u][v]
            base_latency = link_data.get("latency", 1.0)
            congestion_sensitivity = link_data.get("congestion_sensitivity", 1.0)

            if utilization >= 1.0:
                new_latency = base_latency * 1000
            else:
                congestion_delay = congestion_sensitivity * (utilization / (1 - utilization))
                new_latency = base_latency + congestion_delay

            self.current_latencies[(u, v)] = new_latency
            self.current_latencies[(v, u)] = new_latency

    def _get_shortest_path(self, source: int, target: int) -> List[int]:
        """Get shortest path between two nodes"""
        try:
            return nx.shortest_path(self.topology, source, target, weight="latency")
        except nx.NetworkXNoPath:
            return []

    def _track_storage_usage(self):
        """Track storage usage statistics"""
        total_used = sum(usage["used"] for usage in self.storage_usage.values())
        total_capacity = sum(usage["used"] + usage["free"] for usage in self.storage_usage.values())

        if total_capacity > 0:
            usage_percentage = (total_used / total_capacity) * 100
            self.evaluation.record_storage_usage(usage_percentage)

    def _track_congestion_levels(self):
        """Track average congestion levels"""
        if self.link_utilization:
            avg_utilization = np.mean(list(self.link_utilization.values()))
            self.evaluation.record_congestion(avg_utilization)
        else:
            self.evaluation.record_congestion(0.0)

    def run(self, num_steps: Optional[int] = None, verbose: bool = True) -> Dict:
        """Run the simulation for this strategy"""
        if num_steps is not None:
            self.max_steps = num_steps

        self.reset()

        if verbose:
            print(f"Running simulation for strategy: {self.strategy_name}")
            print(f"  Steps: {self.max_steps}")
            print(f"  Congestion: {self.enable_congestion}")
            print(f"  Consistency: {self.enable_consistency}")

        for step in range(self.max_steps):
            self.step()

        results = self.evaluation.get_results()
        return results


# Example usage
if __name__ == "__main__":
    from topology import generate_topology, args
    from placement_strategies import (
        LocalPlacementStrategy,
        RandomReplicaPlacementStrategy,
        PCenterPlacementStrategy
    )

    # Generate topology
    topology, deployment_info, instance_manager = generate_topology(**args)

    # Create three isolated environments
    print("Creating isolated environments for each strategy...\n")

    env_local = DataContinuumEnv(
        topology=topology,
        instance_manager=instance_manager,
        placement_strategy=LocalPlacementStrategy(),
        enable_congestion=True,
        enable_consistency=True,
        random_state=56,
        max_steps=1000
    )

    env_random = DataContinuumEnv(
        topology=topology,
        instance_manager=instance_manager,
        placement_strategy=RandomReplicaPlacementStrategy(num_replicas=3),
        enable_congestion=True,
        enable_consistency=True,
        random_state=56,
        max_steps=1000
    )

    env_pcenter = DataContinuumEnv(
        topology=topology,
        instance_manager=instance_manager,
        placement_strategy=PCenterPlacementStrategy(num_replicas=3),
        enable_congestion=True,
        enable_consistency=True,
        random_state=56,
        max_steps=1000
    )

    # Run each environment independently
    print("\n" + "=" * 80)
    results_local = env_local.run()

    print("\n" + "=" * 80)
    results_random = env_random.run()

    print("\n" + "=" * 80)
    results_pcenter = env_pcenter.run()

    # Print individual results
    print("\n" + "=" * 80)
    print("LOCAL STRATEGY RESULTS")
    print("=" * 80)
    env_local.evaluation.print_results(results_local)

    print("\n" + "=" * 80)
    print("RANDOM STRATEGY RESULTS")
    print("=" * 80)
    env_random.evaluation.print_results(results_random)

    print("\n" + "=" * 80)
    print("P-CENTER STRATEGY RESULTS")
    print("=" * 80)
    env_pcenter.evaluation.print_results(results_pcenter)