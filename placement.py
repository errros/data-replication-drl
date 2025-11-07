"""
SimPy Environment with Leaderless Replication Support
"""
"""
SimPy Environment with Leaderless Replication Support
Allows early reads and tracks read freshness
"""

import simpy
import networkx as nx
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass, field


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class DataItem:
    """Data item with metadata"""
    data_id: str
    generator_instance_id: int
    generator_node: int
    generator_app_name: str
    size_bytes: float
    size_gb: float
    generation_time: float

    # Placement info
    replicas: List[int] = field(default_factory=list)
    target_replicas: int = 0  # How many replicas we're trying to create
    replication_model: str = "leader-follower"  # or "leaderless"
    replication_complete: bool = False
    replication_completion_time: Optional[float] = None

    # Write operation metrics
    write_latency: Optional[float] = None  # Leader: all replicas, Leaderless: first replica
    write_latency_all: Optional[float] = None  # Time until ALL replicas written

    placement_decision_time: Optional[float] = None


@dataclass
class ConsumptionEvent:
    """Record of a data consumption"""
    time: float
    consumer_instance_id: int
    consumer_node: int
    consumer_app_name: str
    generator_app_name: str
    data_id: str
    data_size_bytes: float
    latency: float
    slo: float
    violation: bool
    consistency_mode: int
    num_replicas_read: int
    num_replicas_available: int  # NEW: How many replicas existed at read time
    num_replicas_target: int  # NEW: How many replicas were expected
    early_read: bool  # NEW: Was this an early read? (available < target)


@dataclass
class GenerationEvent:
    """Record of data generation"""
    time: float
    generator_instance_id: int
    generator_node: int
    generator_app_name: str
    data_id: str
    data_size_bytes: float


# ============================================================================
# SimPy Network Model
# ============================================================================

class SimPyLink:
    """Network link modeled as SimPy Resource"""

    def __init__(self, env: simpy.Environment, link_id: Tuple[int, int],
                 bandwidth_mbps: float, latency_ms: float, congestion_sensitivity: float):
        self.env = env
        self.link_id = link_id
        self.bandwidth_bps = bandwidth_mbps * 1e6
        self.latency_sec = latency_ms / 1000.0
        self.congestion_sensitivity = congestion_sensitivity
        self.resource = simpy.Resource(env, capacity=1)

        self.transfer_count = 0
        self.total_bytes_transferred = 0
        self.transfer_log = []

    def transmit(self, data_size_bytes: float, transfer_id: str):
        """Transmit data over link"""
        arrival_time = self.env.now

        with self.resource.request() as request:
            yield request

            queuing_delay = self.env.now - arrival_time
            yield self.env.timeout(self.latency_sec)

            transmission_time = (data_size_bytes * 8) / self.bandwidth_bps
            yield self.env.timeout(transmission_time)

            total_time = queuing_delay + self.latency_sec + transmission_time

            self.transfer_count += 1
            self.total_bytes_transferred += data_size_bytes
            self.transfer_log.append({
                'transfer_id': transfer_id,
                'arrival_time': arrival_time,
                'start_time': arrival_time + queuing_delay,
                'end_time': self.env.now,
                'queuing_delay': queuing_delay,
                'transmission_time': transmission_time,
                'total_time': total_time,
                'data_size': data_size_bytes
            })

            return total_time


class SimPyNetworkModel:
    """Complete network model with SimPy links"""

    def __init__(self, env: simpy.Environment, topology: nx.Graph):
        self.env = env
        self.topology = topology
        self.links: Dict[Tuple[int, int], SimPyLink] = {}

        for (u, v, data) in topology.edges(data=True):
            bandwidth = data.get('bandwidth', 1000)
            latency = data.get('latency', 1.0)
            congestion_sensitivity = data.get('congestion_sensitivity', 0.1)

            self.links[(u, v)] = SimPyLink(env, (u, v), bandwidth, latency, congestion_sensitivity)
            self.links[(v, u)] = SimPyLink(env, (v, u), bandwidth, latency, congestion_sensitivity)

    def transfer_data(self, source: int, target: int, data_size_bytes: float, transfer_id: str):
        """Multi-hop data transfer"""
        if source == target:
            return 0.0

        try:
            path = nx.shortest_path(self.topology, source, target, weight='latency')
        except nx.NetworkXNoPath:
            return float('inf')

        total_latency = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            link = self.links[(u, v)]
            hop_latency = yield self.env.process(
                link.transmit(data_size_bytes, f"{transfer_id}_hop{i}_{u}->{v}")
            )
            total_latency += hop_latency

        return total_latency


# ============================================================================
# Storage & Catalogue
# ============================================================================

class StorageManager:
    """Manages storage on all nodes"""

    def __init__(self, topology: nx.Graph):
        self.storage: Dict[int, Dict] = {}

        for node_id, node_data in topology.nodes(data=True):
            if node_data.get('type') in ['cloud', 'fog', 'edge']:
                capacity = node_data.get('storage_capacity', 0)
                self.storage[node_id] = {
                    'capacity': capacity,
                    'used': 0.0,
                    'free': capacity,
                    'stored_data': set()
                }

    def has_space(self, node_id: int, size_gb: float) -> bool:
        if node_id not in self.storage:
            return False
        return self.storage[node_id]['free'] >= size_gb

    def store_data(self, node_id: int, data_id: str, size_gb: float) -> bool:
        if not self.has_space(node_id, size_gb):
            return False

        self.storage[node_id]['used'] += size_gb
        self.storage[node_id]['free'] -= size_gb
        self.storage[node_id]['stored_data'].add(data_id)
        return True

    def remove_data(self, node_id: int, data_id: str, size_gb: float):
        if node_id in self.storage and data_id in self.storage[node_id]['stored_data']:
            self.storage[node_id]['used'] -= size_gb
            self.storage[node_id]['free'] += size_gb
            self.storage[node_id]['stored_data'].remove(data_id)

    def get_available_nodes(self, min_space_gb: float) -> List[int]:
        return [
            node_id for node_id, info in self.storage.items()
            if info['free'] >= min_space_gb
        ]

    def get_state(self) -> Dict:
        total_used = sum(s['used'] for s in self.storage.values())
        total_capacity = sum(s['capacity'] for s in self.storage.values())

        return {
            'total_used': total_used,
            'total_capacity': total_capacity,
            'usage_percentage': (total_used / total_capacity * 100) if total_capacity > 0 else 0,
            'per_node': {node_id: info.copy() for node_id, info in self.storage.items()}
        }


class GenerationCatalogue:
    """Manages all generated data items"""

    def __init__(self):
        self.data_by_id: Dict[str, DataItem] = {}
        self.data_by_generator: Dict[int, List[DataItem]] = defaultdict(list)

    def add_data(self, data: DataItem):
        self.data_by_id[data.data_id] = data
        self.data_by_generator[data.generator_instance_id].append(data)

    def get_latest_from_generator(self, generator_instance_id: int) -> Optional[DataItem]:
        data_list = self.data_by_generator.get(generator_instance_id, [])
        return data_list[-1] if data_list else None

    def get_data(self, data_id: str) -> Optional[DataItem]:
        return self.data_by_id.get(data_id)


# ============================================================================
# SimPy-Gym Bridge with Leaderless Support
# ============================================================================

class SimPyGymBridge:
    """Bridge between SimPy simulation and Gym environment"""

    def __init__(self, env: simpy.Environment, topology: nx.Graph,
                 instance_manager, latency_cache,
                 replication_model: str = "leader-follower",
                 enable_consistency: bool = True,
                 max_consumptions:Optional[int] = None
                ):
        self.env = env
        self.topology = topology
        self.instance_manager = instance_manager
        self.latency_cache = latency_cache
        self.replication_model = replication_model  # SYSTEM-WIDE setting
        self.enable_consistency = enable_consistency

        self.network = SimPyNetworkModel(env, topology)
        self.storage = StorageManager(topology)
        self.catalogue = GenerationCatalogue()

        self.pending_decisions: deque[DataItem] = deque()
        self.decision_events: Dict[str, simpy.Event] = {}

        self.generation_log: List[GenerationEvent] = []
        self.consumption_log: List[ConsumptionEvent] = []

        self.max_consumptions = max_consumptions
        self.total_consumptions = 0  # Track consumption count
        self.next_data_id = 1

        self._start_processes()

    def _start_processes(self):
        for gen_id in self.instance_manager.generator_instances:
            gen_instance = self.instance_manager.get_instance(gen_id)
            self.env.process(self._generator_process(gen_instance))

        for cons_id in self.instance_manager.consumer_instances:
            cons_instance = self.instance_manager.get_instance(cons_id)
            self.env.process(self._consumer_process(cons_instance))

    def _generator_process(self, instance: Dict):
        """Generator process - uses system-wide replication model"""
        while True:
            yield self.env.timeout(instance['generation_rate'])

            data_id = f"data_{self.next_data_id}"
            self.next_data_id += 1


            data = DataItem(
                data_id=data_id,
                generator_instance_id=instance['instance_id'],
                generator_node=instance['node_id'],
                generator_app_name=instance['name'],
                size_bytes=instance['generated_size'] * 1e9,
                size_gb=instance['generated_size'],
                generation_time=self.env.now
            )

            self.catalogue.add_data(data)

            self.generation_log.append(GenerationEvent(
                time=self.env.now,
                generator_instance_id=instance['instance_id'],
                generator_node=instance['node_id'],
                generator_app_name=instance['name'],
                data_id=data_id,
                data_size_bytes=data.size_bytes
            ))

            self.pending_decisions.append(data)

            decision_event = self.env.event()
            self.decision_events[data_id] = decision_event

            # Wait for placement decision (just nodes list)
            placement_nodes = yield decision_event

            # Use system-wide replication model
            data.replication_model = self.replication_model
            data.target_replicas = len(placement_nodes)

            leaderless = (self.replication_model == "leaderless")
            self.env.process(self._replicate_data(data, placement_nodes, leaderless))

    def _consumer_process(self, instance: Dict):
        """Consumer process - allows early reads"""
        while True:

            # Check if we've reached consumption limit
            if self.max_consumptions is not None and self.total_consumptions >= self.max_consumptions:
                break
            yield self.env.timeout(instance['consumption_rate'])
            for gen_id in instance['connected_generators']:
                latest_data = self.catalogue.get_latest_from_generator(gen_id)

                # EARLY READ SUPPORT: Only check if replication_complete
                # Don't wait for all replicas!
                if latest_data and latest_data.replication_complete:
                    latency = yield self.env.process(
                        self._consume_data(instance, latest_data)
                    )

                    if latency is not None and latency < float('inf'):
                        # Increment consumption counter
                        self.total_consumptions += 1
                        # print(f"CONS N={self.total_consumptions}")
                        # Track read freshness
                        num_available = len(latest_data.replicas)
                        num_target = latest_data.target_replicas
                        consistency_mode = instance['consistency']

                        # IMPORTANT: Only consider it "stale/early" if:
                        # 1. Strong consistency (level 1) AND
                        # 2. Not all replicas available
                        # For weak consistency (level 0), doesn't matter!
                        early_read = (consistency_mode == 1 and num_available < num_target)



                        self.consumption_log.append(ConsumptionEvent(
                            time=self.env.now,
                            consumer_instance_id=instance['instance_id'],
                            consumer_node=instance['node_id'],
                            consumer_app_name=instance['name'],
                            generator_app_name=latest_data.generator_app_name,
                            data_id=latest_data.data_id,
                            data_size_bytes=latest_data.size_bytes,
                            latency=latency * 1000,
                            slo=instance['slo'],
                            violation=latency * 1000 > instance['slo'],
                            consistency_mode=consistency_mode,
                            num_replicas_read=num_available,
                            num_replicas_available=num_available,  # Snapshot at read time
                            num_replicas_target=num_target,
                            early_read=early_read
                        ))

                        if self.max_consumptions is not None and self.total_consumptions >= self.max_consumptions:
                            break

    def _consume_data(self, consumer_instance: Dict, data: DataItem):
        """Read from replicas - reads whatever is available"""
        consumer_node = consumer_instance['node_id']

        if not data.replicas:
            return None

        consistency_mode = consumer_instance['consistency']

        # Snapshot current replicas (might be growing if leaderless)
        available_replicas = list(data.replicas)

        if self.enable_consistency and consistency_mode == 1:
            # Strong: read from ALL available replicas
            read_processes = []
            for idx, replica_node in enumerate(available_replicas):
                read_proc = self.env.process(
                    self.network.transfer_data(
                        replica_node, consumer_node, data.size_bytes,
                        f"consume_{data.data_id}_strong_{idx}"
                    )
                )
                read_processes.append(read_proc)

            yield simpy.AllOf(self.env, read_processes)
            latencies = [proc.value for proc in read_processes]
            return max(latencies) if latencies else None
        else:
            # Weak: read from closest (first to respond)
            read_processes = []
            for idx, replica_node in enumerate(available_replicas):
                read_proc = self.env.process(
                    self.network.transfer_data(
                        replica_node, consumer_node, data.size_bytes,
                        f"consume_{data.data_id}_weak_{idx}"
                    )
                )
                read_processes.append(read_proc)

            yield simpy.AnyOf(self.env, read_processes)
            completed = [proc for proc in read_processes if proc.processed]
            latencies = [proc.value for proc in completed]
            return min(latencies) if latencies else None

    def _replicate_data(self, data: DataItem, target_nodes: List[int], leaderless: bool = False):
        """Replicate data - leader-follower or leaderless"""
        if not target_nodes:
            return

        if leaderless:
            # LEADERLESS: Parallel writes, complete on FIRST
            write_processes = []

            for target_node in target_nodes:
                proc = self.env.process(
                    self._write_to_node(data, data.generator_node, target_node)
                )
                write_processes.append(proc)

            # Wait for FIRST to complete
            yield simpy.AnyOf(self.env, write_processes)

            # Mark available for reads
            data.replication_complete = True
            data.replication_completion_time = self.env.now
            data.write_latency = self.env.now - data.generation_time

            # Monitor background completion
            self.env.process(self._monitor_all_replicas(data, write_processes))

        else:
            # LEADER-FOLLOWER: Sequential primary, then parallel secondaries
            primary_node = self._find_closest_node(data.generator_node, target_nodes)
            secondary_nodes = [n for n in target_nodes if n != primary_node]

            # Write to primary
            if primary_node == data.generator_node:
                if self.storage.store_data(primary_node, data.data_id, data.size_gb):
                    data.replicas.append(primary_node)
            else:
                yield self.env.process(
                    self.network.transfer_data(
                        data.generator_node, primary_node, data.size_bytes,
                        f"write_primary_{data.data_id}"
                    )
                )
                if self.storage.store_data(primary_node, data.data_id, data.size_gb):
                    data.replicas.append(primary_node)

            # Sync to secondaries
            sync_processes = []
            for target_node in secondary_nodes:
                proc = self.env.process(
                    self._synchronize_replica(data, primary_node, target_node)
                )
                sync_processes.append(proc)

            if sync_processes:
                yield simpy.AllOf(self.env, sync_processes)

            # Mark complete
            data.replication_complete = True
            data.replication_completion_time = self.env.now
            data.write_latency = self.env.now - data.generation_time
            data.write_latency_all = data.write_latency  # Same for leader-follower

    def _write_to_node(self, data: DataItem, source_node: int, target_node: int):
        """Write to a single node (for leaderless)"""
        if source_node == target_node:
            if self.storage.store_data(target_node, data.data_id, data.size_gb):
                data.replicas.append(target_node)
            return 0.0
        else:
            transfer_latency = yield self.env.process(
                self.network.transfer_data(
                    source_node, target_node, data.size_bytes,
                    f"leaderless_write_{data.data_id}_{target_node}"
                )
            )

            if self.storage.store_data(target_node, data.data_id, data.size_gb):
                data.replicas.append(target_node)

            return transfer_latency

    def _synchronize_replica(self, data: DataItem, source_node: int, target_node: int):
        """Synchronize from primary to secondary (for leader-follower)"""
        yield self.env.process(
            self.network.transfer_data(
                source_node, target_node, data.size_bytes,
                f"sync_{data.data_id}_{source_node}_to_{target_node}"
            )
        )

        if self.storage.store_data(target_node, data.data_id, data.size_gb):
            data.replicas.append(target_node)

    def _monitor_all_replicas(self, data: DataItem, write_processes: List):
        """Background: track when ALL replicas complete (leaderless only)"""
        all_procs = write_processes
        yield simpy.AllOf(self.env, all_procs)

        data.write_latency_all = self.env.now - data.generation_time

    def _find_closest_node(self, source: int, candidates: List[int]) -> int:
        """Find closest node using latency cache"""
        if not self.latency_cache or source in candidates:
            return candidates[0] if source not in candidates else source

        min_latency = float('inf')
        closest = candidates[0]

        for node in candidates:
            latency = self.latency_cache.get_latency(source, node)
            if latency < min_latency:
                min_latency = latency
                closest = node

        return closest

    def execute_placement_decision(self, data_id: str, placement_nodes: List[int]):
        """Execute placement decision - just pass the node list"""
        if data_id in self.decision_events:
            event = self.decision_events[data_id]
            event.succeed(placement_nodes)
            del self.decision_events[data_id]

    def get_metrics(self, start_time: float = 0.0, end_time: float = None) -> Dict:
        """Get metrics with replication model breakdown"""
        if end_time is None:
            end_time = self.env.now

        consumptions = [
            c for c in self.consumption_log
            if start_time <= c.time < end_time
        ]

        if consumptions:
            latencies = [c.latency for c in consumptions]
            violations = sum(1 for c in consumptions if c.violation)

            # Early read statistics
            early_reads = sum(1 for c in consumptions if c.early_read)

            strong_consumptions = [c for c in consumptions if c.consistency_mode == 1]
            weak_consumptions = [c for c in consumptions if c.consistency_mode == 0]

            metrics = {
                'avg_latency': np.mean(latencies),
                'median_latency': np.median(latencies),
                'p95_latency': np.percentile(latencies, 95),
                'slo_violations': violations,
                'total_consumptions': len(consumptions),
                'slo_violation_rate': violations / len(consumptions),
                'early_reads': early_reads,
                'early_read_rate': early_reads / len(consumptions)
            }

            if strong_consumptions:
                strong_latencies = [c.latency for c in strong_consumptions]
                strong_violations = sum(1 for c in strong_consumptions if c.violation)
                metrics['strong_consistency'] = {
                    'avg_latency': np.mean(strong_latencies),
                    'median_latency': np.median(strong_latencies),
                    'p95_latency': np.percentile(strong_latencies, 95),
                    'slo_violations': strong_violations,
                    'total_consumptions': len(strong_consumptions),
                    'slo_violation_rate': strong_violations / len(strong_consumptions)
                }

            if weak_consumptions:
                weak_latencies = [c.latency for c in weak_consumptions]
                weak_violations = sum(1 for c in weak_consumptions if c.violation)
                metrics['weak_consistency'] = {
                    'avg_latency': np.mean(weak_latencies),
                    'median_latency': np.median(weak_latencies),
                    'p95_latency': np.percentile(weak_latencies, 95),
                    'slo_violations': weak_violations,
                    'total_consumptions': len(weak_consumptions),
                    'slo_violation_rate': weak_violations / len(weak_consumptions)
                }

            metrics['by_application'] = self._get_application_metrics(consumptions)

        else:
            metrics = {
                'avg_latency': 0.0,
                'median_latency': 0.0,
                'p95_latency': 0.0,
                'slo_violations': 0,
                'total_consumptions': 0,
                'slo_violation_rate': 0.0,
                'early_reads': 0,
                'early_read_rate': 0.0,
                'by_application': {}
            }

        # Write latency metrics by replication model
        generations = [g for g in self.generation_log if start_time <= g.time < end_time]

        leader_data = []
        leaderless_data = []

        for gen_event in generations:
            data = self.catalogue.get_data(gen_event.data_id)
            if data and data.write_latency is not None:
                if data.replication_model == "leaderless":
                    leaderless_data.append(data)
                else:
                    leader_data.append(data)

        if leader_data:
            leader_latencies = [d.write_latency for d in leader_data]
            metrics['leader_follower_write'] = {
                'avg': np.mean(leader_latencies),
                'median': np.median(leader_latencies),
                'p95': np.percentile(leader_latencies, 95),
                'count': len(leader_data)
            }

        if leaderless_data:
            first_latencies = [d.write_latency for d in leaderless_data]
            all_latencies = [d.write_latency_all for d in leaderless_data if d.write_latency_all]

            metrics['leaderless_write'] = {
                'first_avg': np.mean(first_latencies),
                'first_p95': np.percentile(first_latencies, 95),
                'all_avg': np.mean(all_latencies) if all_latencies else None,
                'all_p95': np.percentile(all_latencies, 95) if all_latencies else None,
                'count': len(leaderless_data)
            }

        # Storage and network
        storage_state = self.storage.get_state()
        metrics['storage_usage_pct'] = storage_state['usage_percentage']

        total_bytes = sum(link.total_bytes_transferred for link in self.network.links.values())
        metrics['total_bytes_transferred'] = total_bytes

        metrics['data_generated'] = len(generations)

        return metrics

    def _get_application_metrics(self, consumptions: List[ConsumptionEvent]) -> Dict:
        """Get metrics broken down by application type"""
        app_metrics = {}

        by_gen_app = defaultdict(list)
        for c in consumptions:
            by_gen_app[c.generator_app_name].append(c)

        for app_name, app_consumptions in by_gen_app.items():
            strong = [c for c in app_consumptions if c.consistency_mode == 1]
            weak = [c for c in app_consumptions if c.consistency_mode == 0]

            app_metrics[app_name] = {
                'total_consumptions': len(app_consumptions),
                'strong_consistency': self._compute_consistency_metrics(strong) if strong else None,
                'weak_consistency': self._compute_consistency_metrics(weak) if weak else None
            }

        return app_metrics

    def _compute_consistency_metrics(self, consumptions: List[ConsumptionEvent]) -> Dict:
        """Compute metrics for a specific subset of consumptions"""
        if not consumptions:
            return None

        latencies = [c.latency for c in consumptions]
        violations = sum(1 for c in consumptions if c.violation)

        return {
            'avg_latency': np.mean(latencies),
            'median_latency': np.median(latencies),
            'p95_latency': np.percentile(latencies, 95),
            'slo_violations': violations,
            'total_consumptions': len(consumptions),
            'slo_violation_rate': violations / len(consumptions)
        }
"""
Compare Random-3 Placement with Leader-Follower vs Leaderless Replication
"""

import simpy
import random
import numpy as np
from topology import args, generate_topology, ShortestPathCache

from placement_strategies import RandomReplicaPlacementStrategy, PlacementContext


def run_random_strategy(bridge, strategy, env, latency_cache):
    """Controller using Random-3 placement strategy"""
    rng = random.Random(42)

    while True:
        # Wait for pending decisions
        while not bridge.pending_decisions:
            yield env.timeout(0.1)

        # Get pending data
        data = bridge.pending_decisions.popleft()

        # Create placement context
        context = PlacementContext(
            topology=bridge.topology,
            instance_manager=bridge.instance_manager,
            latency_cache=latency_cache,
            storage_usage=bridge.storage.storage,
            rng=rng
        )

        # Convert DataItem to dict format expected by strategy
        data_item = {
            'id': data.data_id,
            'generator_node': data.generator_node,
            'generator_instance': data.generator_instance_id,
            'size_gb': data.size_gb,
            'size_bytes': data.size_bytes
        }

        # Apply placement strategy
        placement = strategy.place(data.data_id, data_item, context)

        # Execute placement (just nodes, model is system-wide)
        bridge.execute_placement_decision(data.data_id, placement)


def compare_replication_models():
    """Compare Random-3 placement with both replication models"""

    print("=" * 80)
    print("REPLICATION MODEL COMPARISON: Random-3 Placement")
    print("=" * 80)

    # Generate topology (same for both)
    topology, infos, instance_manager = generate_topology(**args)
    cache = ShortestPathCache(topology)

    print(f"\nTopology: {len(topology.nodes)} nodes, {len(topology.edges)} edges")
    print(f"Generators: {len(instance_manager.generator_instances)}")
    print(f"Consumers: {len(instance_manager.consumer_instances)}")

    # Configuration
    simulation_time = 1000
    num_replicas = 3

    results = {}

    # ========================================================================
    # Run 1: Leader-Follower
    # ========================================================================
    print("\n" + "=" * 80)
    print(f"RUNNING LEADER-FOLLOWER ({num_replicas} replicas)")
    print("=" * 80)

    env_leader = simpy.Environment()
    bridge_leader = SimPyGymBridge(
        env=env_leader,
        topology=topology,
        instance_manager=instance_manager,
        latency_cache=cache,
        replication_model="leader-follower",
        enable_consistency=True
    )

    strategy_leader = RandomReplicaPlacementStrategy(num_replicas=num_replicas)

    env_leader.process(run_random_strategy(bridge_leader, strategy_leader, env_leader, cache))
    env_leader.run(until=simulation_time)

    results['leader-follower'] = bridge_leader.get_metrics(start_time=0.0, end_time=env_leader.now)
    print(f"Simulation complete! Time: {env_leader.now:.2f}")

    # ========================================================================
    # Run 2: Leaderless
    # ========================================================================
    print("\n" + "=" * 80)
    print(f"RUNNING LEADERLESS ({num_replicas} replicas)")
    print("=" * 80)

    env_leaderless = simpy.Environment()
    bridge_leaderless = SimPyGymBridge(
        env=env_leaderless,
        topology=topology,
        instance_manager=instance_manager,
        latency_cache=cache,
        replication_model="leaderless",
        enable_consistency=True
    )

    strategy_leaderless = RandomReplicaPlacementStrategy(num_replicas=num_replicas)

    env_leaderless.process(run_random_strategy(bridge_leaderless, strategy_leaderless, env_leaderless, cache))
    env_leaderless.run(until=simulation_time)

    results['leaderless'] = bridge_leaderless.get_metrics(start_time=0.0, end_time=env_leaderless.now)
    print(f"Simulation complete! Time: {env_leaderless.now:.2f}")

    # ========================================================================
    # COMPARISON RESULTS
    # ========================================================================
    print("\n" + "=" * 80)
    print("COMPARISON: LEADER-FOLLOWER vs LEADERLESS (Random-3)")
    print("=" * 80)

    lf = results['leader-follower']
    ll = results['leaderless']

    print(f"\n{'Metric':<40} {'Leader-Follower':<25} {'Leaderless':<25} {'Winner':<15}")
    print("-" * 105)

    # Data throughput
    print(f"{'Data Generated':<40} {lf['data_generated']:<25} {ll['data_generated']:<25}")
    print(f"{'Data Consumed':<40} {lf['total_consumptions']:<25} {ll['total_consumptions']:<25}")

    # Write latency
    print(f"\n{'--- WRITE LATENCY ---':<40}")

    if 'leader_follower_write' in lf:
        lf_write = lf['leader_follower_write']['avg'] * 1000
        print(f"{'Leader-Follower (to all replicas)':<40} {lf_write:<25.2f} {'ms':<25}")

    if 'leaderless_write' in ll:
        ll_write_first = ll['leaderless_write']['first_avg'] * 1000
        ll_write_all = ll['leaderless_write']['all_avg'] * 1000 if ll['leaderless_write']['all_avg'] else 0
        print(f"{'Leaderless (to first replica)':<40} {ll_write_first:<25.2f} {'ms':<25}")
        print(f"{'Leaderless (to all replicas)':<40} {ll_write_all:<25.2f} {'ms':<25}")

        winner_write = "Leaderless" if ll_write_first < lf_write else "Leader-Follower"
        improvement = ((lf_write - ll_write_first) / lf_write) * 100 if lf_write > 0 else 0
        print(f"{'Write Winner (all replicas)':<40} {winner_write:<25} {f'{improvement:+.1f}%':<25}")

    # Read latency
    print(f"\n{'--- READ LATENCY ---':<40}")
    lf_read = lf['avg_latency']
    ll_read = ll['avg_latency']
    winner_read = "Leaderless" if ll_read < lf_read else "Leader-Follower"
    improvement_read = ((lf_read - ll_read) / lf_read) * 100 if lf_read > 0 else 0

    print(f"{'Average Read Latency (ms)':<40} {lf_read:<25.2f} {ll_read:<25.2f} {winner_read:<15}")
    print(f"{'P95 Read Latency (ms)':<40} {lf['p95_latency']:<25.2f} {ll['p95_latency']:<25.2f}")
    print(f"{'Read Latency Improvement':<40} {'':<25} {f'{improvement_read:+.1f}%':<25}")

    # SLO violations
    print(f"\n{'--- SLO VIOLATIONS ---':<40}")
    lf_slo = lf['slo_violation_rate']
    ll_slo = ll['slo_violation_rate']
    winner_slo = "Leaderless" if ll_slo < lf_slo else "Leader-Follower"
    improvement_slo = ((lf_slo - ll_slo) / lf_slo) * 100 if lf_slo > 0 else 0

    print(f"{'SLO Violation Rate':<40} {lf_slo:<25.2%} {ll_slo:<25.2%} {winner_slo:<15}")
    print(f"{'SLO Improvement':<40} {'':<25} {f'{improvement_slo:+.1f}%':<25}")

    # Early reads (only for leaderless)
    print(f"\n{'--- EARLY READS (Staleness) ---':<40}")
    lf_early = lf.get('early_read_rate', 0.0)
    ll_early = ll.get('early_read_rate', 0.0)
    print(f"{'Early Read Rate':<40} {lf_early:<25.2%} {ll_early:<25.2%}")
    print(f"{'Explanation':<40} {'Always complete':<25} {f'{ll_early:.1%} read stale data':<25}")

    # Consistency breakdown
    if 'strong_consistency' in lf and 'strong_consistency' in ll:
        print(f"\n{'--- STRONG CONSISTENCY ---':<40}")
        lf_strong = lf['strong_consistency']['avg_latency']
        ll_strong = ll['strong_consistency']['avg_latency']
        winner_strong = "Leaderless" if ll_strong < lf_strong else "Leader-Follower"
        print(f"{'Avg Latency (ms)':<40} {lf_strong:<25.2f} {ll_strong:<25.2f} {winner_strong:<15}")
        print(
            f"{'SLO Violation Rate':<40} {lf['strong_consistency']['slo_violation_rate']:<25.2%} {ll['strong_consistency']['slo_violation_rate']:<25.2%}")

    if 'weak_consistency' in lf and 'weak_consistency' in ll:
        print(f"\n{'--- WEAK CONSISTENCY ---':<40}")
        lf_weak = lf['weak_consistency']['avg_latency']
        ll_weak = ll['weak_consistency']['avg_latency']
        winner_weak = "Leaderless" if ll_weak < lf_weak else "Leader-Follower"
        print(f"{'Avg Latency (ms)':<40} {lf_weak:<25.2f} {ll_weak:<25.2f} {winner_weak:<15}")
        print(
            f"{'SLO Violation Rate':<40} {lf['weak_consistency']['slo_violation_rate']:<25.2%} {ll['weak_consistency']['slo_violation_rate']:<25.2%}")

    # Resources
    print(f"\n{'--- RESOURCES ---':<40}")
    print(f"{'Storage Usage (%)':<40} {lf['storage_usage_pct']:<25.2f} {ll['storage_usage_pct']:<25.2f}")
    print(
        f"{'Total Data Transferred (GB)':<40} {lf['total_bytes_transferred'] / 1e9:<25.2f} {ll['total_bytes_transferred'] / 1e9:<25.2f}")

    # Application breakdown
    print("\n" + "=" * 80)
    print("APPLICATION-LEVEL COMPARISON")
    print("=" * 80)

    for app_name in lf.get('by_application', {}).keys():
        print(f"\n--- {app_name.upper()} ---")

        lf_app = lf['by_application'][app_name]
        ll_app = ll['by_application'].get(app_name, {})

        if lf_app.get('strong_consistency') and ll_app.get('strong_consistency'):
            lf_strong = lf_app['strong_consistency']['avg_latency']
            ll_strong = ll_app['strong_consistency']['avg_latency']
            winner = "Leaderless" if ll_strong < lf_strong else "Leader-Follower"
            print(f"  STRONG consistency latency:")
            print(f"    Leader-Follower: {lf_strong:>8.2f} ms")
            print(f"    Leaderless:      {ll_strong:>8.2f} ms")
            print(f"    Winner:          {winner}")

        if lf_app.get('weak_consistency') and ll_app.get('weak_consistency'):
            lf_weak = lf_app['weak_consistency']['avg_latency']
            ll_weak = ll_app['weak_consistency']['avg_latency']
            winner = "Leaderless" if ll_weak < lf_weak else "Leader-Follower"
            print(f"  WEAK consistency latency:")
            print(f"    Leader-Follower: {lf_weak:>8.2f} ms")
            print(f"    Leaderless:      {ll_weak:>8.2f} ms")
            print(f"    Winner:          {winner}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: LEADER-FOLLOWER vs LEADERLESS")
    print("=" * 80)

    print("\nLeader-Follower Characteristics:")
    print(f"  ✓ Write latency: {lf_write:.2f}ms (to all replicas)")
    print(f"  ✓ Early reads: {lf_early:.1%} (always complete data)")
    print(f"  ✓ Read latency: {lf_read:.2f}ms")
    print(f"  ✓ SLO violations: {lf_slo:.2%}")
    print(f"  → Best for: Strict consistency, no stale reads acceptable")

    print("\nLeaderless Characteristics:")
    print(f"  ✓ Write latency: {ll_write_first:.2f}ms (to first), {ll_write_all:.2f}ms (to all)")
    print(f"  ✓ Early reads: {ll_early:.1%} (reads before all replicas ready)")
    print(f"  ✓ Read latency: {ll_read:.2f}ms")
    print(f"  ✓ SLO violations: {ll_slo:.2%}")
    print(f"  → Best for: Fast availability, eventual consistency OK")

    print("\nPerformance Improvements (Leaderless vs Leader-Follower):")
    print(f"  Write latency (to all): {improvement:+.1f}%")
    print(f"  Read latency: {improvement_read:+.1f}%")
    print(f"  SLO violations: {improvement_slo:+.1f}%")

    print("\nTrade-offs:")
    if ll_early > 0.1:
        print(f"  ⚠ Leaderless has {ll_early:.1%} early reads (stale data)")
        print(f"    → {ll_early:.1%} of reads may see incomplete replicas")
    else:
        print(f"  ✓ Minimal early reads in leaderless")

    if ll_slo < lf_slo:
        print(f"  ✓ Leaderless reduces SLO violations by {-improvement_slo:.1f}%")
    else:
        print(f"  ⚠ Leader-follower has {improvement_slo:.1f}% fewer SLO violations")

    print("\n" + "=" * 80)
    print("Comparison complete!")
    print("=" * 80)


if __name__ == '__main__':
    compare_replication_models()


