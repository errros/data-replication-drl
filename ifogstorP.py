"""
Compare iFogStorP vs Random-3 Placement
Focus on strong consistency (consistency=1) consumers
"""

import simpy
import random
import numpy as np
from topology import args, generate_topology, ShortestPathCache
from placement import SimPyGymBridge, DataItem, ConsumptionEvent
from placement_strategies import RandomReplicaPlacementStrategy, iFogStorPPlacementStrategy, PlacementContext


def run_placement_strategy(bridge, strategy, env, latency_cache, max_consumptions):
    """Controller using given placement strategy"""
    rng = random.Random(42)

    while bridge.total_consumptions < max_consumptions:
        # Wait for pending decisions
        while not bridge.pending_decisions:
            if bridge.total_consumptions >= max_consumptions:
                return
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

        # Execute placement
        bridge.execute_placement_decision(data.data_id, placement)


def compute_all_consumption_metrics(consumptions: list) -> dict:
    """Compute metrics for ALL consumptions, track stale reads only for strong consistency"""
    if not consumptions:
        return {
            'count': 0,
            'avg_latency': 0,
            'p95_latency': 0,
            'slo_violations': 0,
            'slo_violation_rate': 0,
            'stale_reads': 0,
            'stale_read_rate': 0,
            'strong_consistency_count': 0,
            'weak_consistency_count': 0
        }

    # ALL consumptions
    latencies = [c.latency for c in consumptions]
    violations = sum(1 for c in consumptions if c.violation)

    # Stale reads ONLY for strong consistency
    strong_consumptions = [c for c in consumptions if c.consistency_mode == 1]
    stale_reads = sum(1 for c in strong_consumptions if c.early_read)

    return {
        'count': len(consumptions),
        'avg_latency': np.mean(latencies),
        'p95_latency': np.percentile(latencies, 95),
        'slo_violations': violations,
        'slo_violation_rate': violations / len(consumptions),
        'stale_reads': stale_reads,
        'stale_read_rate': stale_reads / len(strong_consumptions) if strong_consumptions else 0,
        'strong_consistency_count': len(strong_consumptions),
        'weak_consistency_count': len(consumptions) - len(strong_consumptions)
    }


def analyze_placement_decisions(catalogue, strategy_name: str):
    """Analyze what placement decisions were made"""
    replica_counts = []

    for data_id, data in catalogue.data_by_id.items():
        num_replicas = len(data.replicas)
        replica_counts.append(num_replicas)

    if replica_counts:
        print(f"\n{strategy_name} Placement Analysis:")
        print(f"  Total data items: {len(replica_counts)}")
        print(f"  Avg replicas per data: {np.mean(replica_counts):.2f}")
        print(f"  Min replicas: {np.min(replica_counts)}")
        print(f"  Max replicas: {np.max(replica_counts)}")
        print(f"  Replica distribution: {np.bincount(replica_counts)}")

    return {
        'avg_replicas': np.mean(replica_counts) if replica_counts else 0,
        'min_replicas': np.min(replica_counts) if replica_counts else 0,
        'max_replicas': np.max(replica_counts) if replica_counts else 0
    }

def extract_write_latencies(catalogue, replication_model: str) -> dict:
    """Extract write latency statistics"""
    write_latencies = []

    for data_id, data in catalogue.data_by_id.items():
        if data.replication_model == replication_model and data.write_latency is not None:
            write_latencies.append(data.write_latency * 1000)  # Convert to ms

    if not write_latencies:
        return {
            'count': 0,
            'avg': 0,
            'p95': 0
        }

    return {
        'count': len(write_latencies),
        'avg': np.mean(write_latencies),
        'p95': np.percentile(write_latencies, 95)
    }


def compare_strategies():
    """Compare iFogStorP (3-6) vs Random-3"""

    print("=" * 80)
    print("COMPARISON: iFogStorP (p_min=3, p_max=6) vs Random-3")
    print("=" * 80)

    # Generate topology (same for both)
    topology, infos, instance_manager = generate_topology(**args)
    cache = ShortestPathCache(topology)

    print(f"\nTopology: {len(topology.nodes)} nodes, {len(topology.edges)} edges")
    print(f"Generators: {len(instance_manager.generator_instances)}")
    print(f"Consumers: {len(instance_manager.consumer_instances)}")

    # Count strong consistency consumers
    strong_consumers = sum(
        1 for cons_id in instance_manager.consumer_instances
        if instance_manager.get_instance(cons_id)['consistency'] == 1
    )
    print(f"Strong consistency consumers: {strong_consumers}")

    # Configuration
    max_consumptions = 10000
    replication_model = "leaderless"

    results = {}

    # ========================================================================
    # Run 1: Random-3
    # ========================================================================
    print("\n" + "=" * 80)
    print("RUNNING RANDOM-3")
    print("=" * 80)

    env_random = simpy.Environment()
    bridge_random = SimPyGymBridge(
        env=env_random,
        topology=topology,
        instance_manager=instance_manager,
        latency_cache=cache,
        replication_model=replication_model,
        enable_consistency=True,
        max_consumptions=max_consumptions
    )

    strategy_random = RandomReplicaPlacementStrategy(num_replicas=3)

    env_random.process(run_placement_strategy(
        bridge_random, strategy_random, env_random, cache, max_consumptions
    ))
    env_random.run()

    print(f"Simulation complete!")
    print(f"  Total consumptions: {bridge_random.total_consumptions}")
    print(f"  Simulation time: {env_random.now:.2f}")

    placement_analysis_random = analyze_placement_decisions(bridge_random.catalogue, "Random-3")


    # Extract metrics
    write_metrics_random = extract_write_latencies(bridge_random.catalogue, replication_model)
    read_metrics_random = compute_all_consumption_metrics(bridge_random.consumption_log)


    results['random'] = {
        'write': write_metrics_random,
        'read': read_metrics_random
    }

    # ========================================================================
    # Run 2: iFogStorP
    # ========================================================================
    print("\n" + "=" * 80)
    print("RUNNING iFogStorP (p_min=3, p_max=6)")
    print("=" * 80)

    env_ifogstorp = simpy.Environment()
    bridge_ifogstorp = SimPyGymBridge(
        env=env_ifogstorp,
        topology=topology,
        instance_manager=instance_manager,
        latency_cache=cache,
        replication_model=replication_model,
        enable_consistency=True,
        max_consumptions=max_consumptions
    )

    strategy_ifogstorp = iFogStorPPlacementStrategy(p_min=2, p_max=6)

    env_ifogstorp.process(run_placement_strategy(
        bridge_ifogstorp, strategy_ifogstorp, env_ifogstorp, cache, max_consumptions
    ))
    env_ifogstorp.run()

    print(f"Simulation complete!")
    print(f"  Total consumptions: {bridge_ifogstorp.total_consumptions}")
    print(f"  Simulation time: {env_ifogstorp.now:.2f}")
    placement_analysis_ifogstorp = analyze_placement_decisions(bridge_ifogstorp.catalogue, "iFogStorP")
    # Extract metrics
    write_metrics_ifogstorp = extract_write_latencies(bridge_ifogstorp.catalogue, replication_model)
    read_metrics_ifogstorp = compute_all_consumption_metrics(bridge_ifogstorp.consumption_log)

    results['ifogstorp'] = {
        'write': write_metrics_ifogstorp,
        'read': read_metrics_ifogstorp
    }

    # ========================================================================
    # COMPARISON RESULTS
    # ========================================================================
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS (All Consumptions)")
    print("=" * 80)

    random_res = results['random']
    ifogstorp_res = results['ifogstorp']

    print(f"\n{'Metric':<40} {'Random-3':<20} {'iFogStorP':<20} {'Winner':<15}")
    print("-" * 95)

    # Write latency
    print(f"\n{'--- WRITE LATENCY ---':<40}")

    rand_write_avg = random_res['write']['avg']
    ifog_write_avg = ifogstorp_res['write']['avg']
    winner_write_avg = "iFogStorP" if ifog_write_avg < rand_write_avg else "Random-3"
    improvement_write_avg = ((rand_write_avg - ifog_write_avg) / rand_write_avg) * 100 if rand_write_avg > 0 else 0

    print(f"{'Avg Write Latency (ms)':<40} {rand_write_avg:<20.2f} {ifog_write_avg:<20.2f} {winner_write_avg:<15}")

    rand_write_p95 = random_res['write']['p95']
    ifog_write_p95 = ifogstorp_res['write']['p95']
    winner_write_p95 = "iFogStorP" if ifog_write_p95 < rand_write_p95 else "Random-3"
    improvement_write_p95 = ((rand_write_p95 - ifog_write_p95) / rand_write_p95) * 100 if rand_write_p95 > 0 else 0

    print(f"{'P95 Write Latency (ms)':<40} {rand_write_p95:<20.2f} {ifog_write_p95:<20.2f} {winner_write_p95:<15}")
    print(f"{'Write Improvement':<40} {'':<20} {f'{improvement_write_avg:+.1f}%':<20}")

    # Read latency
    print(f"\n{'--- READ LATENCY (Strong Consistency) ---':<40}")

    rand_read_avg = random_res['read']['avg_latency']
    ifog_read_avg = ifogstorp_res['read']['avg_latency']
    winner_read_avg = "iFogStorP" if ifog_read_avg < rand_read_avg else "Random-3"
    improvement_read_avg = ((rand_read_avg - ifog_read_avg) / rand_read_avg) * 100 if rand_read_avg > 0 else 0

    print(f"{'Avg Read Latency (ms)':<40} {rand_read_avg:<20.2f} {ifog_read_avg:<20.2f} {winner_read_avg:<15}")

    rand_read_p95 = random_res['read']['p95_latency']
    ifog_read_p95 = ifogstorp_res['read']['p95_latency']
    winner_read_p95 = "iFogStorP" if ifog_read_p95 < rand_read_p95 else "Random-3"
    improvement_read_p95 = ((rand_read_p95 - ifog_read_p95) / rand_read_p95) * 100 if rand_read_p95 > 0 else 0

    print(f"{'P95 Read Latency (ms)':<40} {rand_read_p95:<20.2f} {ifog_read_p95:<20.2f} {winner_read_p95:<15}")
    print(f"{'Read Improvement':<40} {'':<20} {f'{improvement_read_avg:+.1f}%':<20}")

    # SLO violations
    print(f"\n{'--- SLO VIOLATIONS (Strong Consistency) ---':<40}")

    rand_slo = random_res['read']['slo_violation_rate']
    ifog_slo = ifogstorp_res['read']['slo_violation_rate']
    winner_slo = "iFogStorP" if ifog_slo < rand_slo else "Random-3"
    improvement_slo = ((rand_slo - ifog_slo) / rand_slo) * 100 if rand_slo > 0 else 0

    print(f"{'SLO Violation Rate':<40} {rand_slo:<20.2%} {ifog_slo:<20.2%} {winner_slo:<15}")
    print(f"{'SLO Improvement':<40} {'':<20} {f'{improvement_slo:+.1f}%':<20}")

    # Stale reads
    print(f"\n{'--- STALE READS (Strong Consistency) ---':<40}")

    rand_stale = random_res['read']['stale_read_rate']
    ifog_stale = ifogstorp_res['read']['stale_read_rate']
    winner_stale = "iFogStorP" if ifog_stale < rand_stale else "Random-3"

    print(f"{'Stale Read Rate':<40} {rand_stale:<20.2%} {ifog_stale:<20.2%} {winner_stale:<15}")
    print(
        f"{'Total Stale Reads':<40} {random_res['read']['stale_reads']:<20} {ifogstorp_res['read']['stale_reads']:<20}")

    # Data throughput
    print(f"\n{'--- DATA THROUGHPUT ---':<40}")
    print(f"{'Consumptions (Strong)':<40} {random_res['read']['count']:<20} {ifogstorp_res['read']['count']:<20}")
    print(f"{'Write Operations':<40} {random_res['write']['count']:<20} {ifogstorp_res['write']['count']:<20}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\nRandom-3:")
    print(f"  Write latency: {rand_write_avg:.2f}ms (avg), {rand_write_p95:.2f}ms (p95)")
    print(f"  Read latency: {rand_read_avg:.2f}ms (avg), {rand_read_p95:.2f}ms (p95)")
    print(f"  SLO violations: {rand_slo:.2%}")
    print(f"  Stale reads: {rand_stale:.2%}")

    print("\niFogStorP (3-6):")
    print(f"  Write latency: {ifog_write_avg:.2f}ms (avg), {ifog_write_p95:.2f}ms (p95)")
    print(f"  Read latency: {ifog_read_avg:.2f}ms (avg), {ifog_read_p95:.2f}ms (p95)")
    print(f"  SLO violations: {ifog_slo:.2%}")
    print(f"  Stale reads: {ifog_stale:.2%}")

    print("\nPerformance Improvements (iFogStorP vs Random-3):")
    print(f"  Avg write latency: {improvement_write_avg:+.1f}%")
    print(f"  P95 write latency: {improvement_write_p95:+.1f}%")
    print(f"  Avg read latency: {improvement_read_avg:+.1f}%")
    print(f"  P95 read latency: {improvement_read_p95:+.1f}%")
    print(f"  SLO violations: {improvement_slo:+.1f}%")

    print("\n" + "=" * 80)
    print("Comparison complete!")
    print("=" * 80)


if __name__ == '__main__':
    compare_strategies()