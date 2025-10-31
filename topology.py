from __future__ import annotations

import random
from typing import Dict, List, Tuple, Optional
import networkx as nx
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
args = {
    "random_state": 56,

    "num_fogs": 4,
    "num_edges": 14,
    "terminals_per_edge_interval": [10, 30],

    # Storage capacities (GB)
    "cloud_storage_interval": [10000, 50000],
    "fog_storage_interval": [1000, 5000],
    "edge_storage_interval": [100, 500],

    # Link bandwidths (Mbps)
    "bandwidth_cloud_fog_interval": [1000, 10000],
    "bandwidth_fog_edge_interval": [500, 1000],
    "bandwidth_edge_terminal_interval": [100, 1000],

    # Latencies (ms)
    "latency_cloud_fog_interval": [5, 10],
    "latency_fog_edge_interval": [5, 20],
    "latency_edge_terminal_interval": [1, 5],

    # Congestion sensitivity
    "congestion_sensitivity_cloud_interval": [0.05, 0.1],
    "congestion_sensitivity_fog_interval": [0.1, 0.2],
    "congestion_sensitivity_edge_interval": [0.2, 0.4],

    # Producers
    "generator_apps": [
        {
            "name": "cam_stream",
            "consistency": 1,
            "generation_rate_interval": [30, 60],
            "generated_size_interval": [0.001, 0.01],
            "terminals_deployment_share": 0.1,
            "storage_nodes_deployment_share": 0.0
        },
        {
            "name": "sensor_data",
            "consistency": 0,
            "generation_rate_interval": [10, 30],
            "generated_size_interval": [0.0001, 0.001],
            "terminals_deployment_share": 0.1,
            "storage_nodes_deployment_share": 0.0
        },
        {
            "name": "iot_telemetry",
            "consistency": 0,
            "generation_rate_interval": [120, 300],
            "generated_size_interval": [0.00001, 0.0001],
            "terminals_deployment_share": 0,
            "storage_nodes_deployment_share": 0.15
        }
    ],
    # Consumers

    "consumer_apps": [
        {
            "name": "video_analytics",
            "consistency": 1,
            "consumption_rate_interval": [120, 300],
            "slo": 10,
            "terminals_deployment_share": 0.0,
            "storage_nodes_deployment_share": 0.2
        },
        {
            "name": "batch_processing",
            "consistency": 1,
            "consumption_rate_interval": [300, 400],
            "slo": 10,
            "terminals_deployment_share": 0.0,
            "storage_nodes_deployment_share": 0.1
        },
        {
            "name": "real_time_monitor",
            "consistency": 0,
            "consumption_rate_interval": [500, 700],
            "slo": 50,
            "terminals_deployment_share": 0.1,
            "storage_nodes_deployment_share": 0.1
        }
    ]

}

# -----------------------------
# Utilities
# -----------------------------
def _rng(seed: Optional[int]) -> random.Random:
    r = random.Random()
    if seed is not None:
        r.seed(seed)
    return r

def _draw_from_interval(
        interval: Tuple[float, float],
        r: random.Random,
        integer: bool = False
) -> float:
    a, b = interval
    if integer:
        return r.randint(int(a), int(b))
    return r.uniform(a, b)

# -----------------------------
# Application Instance Management
# -----------------------------
class ApplicationInstanceManager:
    def __init__(self):
        self.next_instance_id = 1
        self.all_instances = {}
        self.generator_instances = []
        self.consumer_instances = []

    def create_generator_instance(self, app_config: Dict, node_id: int, r: random.Random) -> Dict:
        instance_id = self.next_instance_id
        self.next_instance_id += 1

        instance = {
            "instance_id": instance_id,
            "node_id": node_id,
            "type": "generator",
            "name": app_config["name"],
            "consistency": app_config["consistency"],
            "generation_rate": _draw_from_interval(app_config["generation_rate_interval"], r, integer=True),
            "generated_size": _draw_from_interval(app_config["generated_size_interval"], r),
            "app_config": app_config,
            "connected_consumers": []
        }

        self.all_instances[instance_id] = instance
        self.generator_instances.append(instance_id)
        return instance

    def create_consumer_instance(self, app_config: Dict, node_id: int, r: random.Random) -> Dict:
        instance_id = self.next_instance_id
        self.next_instance_id += 1

        instance = {
            "instance_id": instance_id,
            "node_id": node_id,
            "type": "consumer",
            "name": app_config["name"],
            "consistency": app_config["consistency"],
            "consumption_rate": _draw_from_interval(app_config["consumption_rate_interval"], r, integer=True),
            "slo": app_config["slo"],
            "app_config": app_config,
            "connected_generators": []
        }

        self.all_instances[instance_id] = instance
        self.consumer_instances.append(instance_id)
        return instance

    def connect_generator_to_consumers(self, generator_instance_id: int, num_consumers: int = 3, r: random.Random = None):
        if not self.consumer_instances or num_consumers <= 0:
            return

        generator_instance = self.all_instances[generator_instance_id]

        available_consumers = [
            cons_id for cons_id in self.consumer_instances
            if self.all_instances[cons_id]["node_id"] != generator_instance["node_id"]
        ]

        if len(available_consumers) < num_consumers:
            available_consumers = self.consumer_instances.copy()

        selected_consumers = r.sample(available_consumers, min(num_consumers, len(available_consumers)))
        generator_instance["connected_consumers"] = selected_consumers

        for consumer_id in selected_consumers:
            consumer_instance = self.all_instances[consumer_id]
            consumer_instance["connected_generators"].append(generator_instance_id)

    def get_instance(self, instance_id: int) -> Optional[Dict]:
        return self.all_instances.get(instance_id)

    def get_generator_instances_on_node(self, node_id: int) -> List[Dict]:
        return [self.all_instances[inst_id] for inst_id in self.generator_instances
                if self.all_instances[inst_id]["node_id"] == node_id]

    def get_consumer_instances_on_node(self, node_id: int) -> List[Dict]:
        return [self.all_instances[inst_id] for inst_id in self.consumer_instances
                if self.all_instances[inst_id]["node_id"] == node_id]

# -----------------------------
# Attribute generators
# -----------------------------
def make_storage_node_attrs(
        node_type: str,
        r: random.Random,
        storage_interval: Tuple[int, int]
) -> Dict:
    storage_capacity = _draw_from_interval(storage_interval, r, integer=True)

    return {
        "type": node_type,
        "storage_capacity": storage_capacity,
        "free_space": storage_capacity,
        "stored_data_ids": [],
        "generator_instances": [],
        "consumer_instances": []
    }

def make_terminal_node_attrs() -> Dict:
    return {
        "type": "terminal",
        "generator_instances": [],
        "consumer_instances": []
    }

def make_link_attrs(
        r: random.Random,
        bandwidth_interval: Tuple[float, float],
        latency_interval: Tuple[float, float],
        congestion_sensitivity_interval: Tuple[float, float]
) -> Dict:
    return {
        "bandwidth": _draw_from_interval(bandwidth_interval, r),
        "latency": _draw_from_interval(latency_interval, r),
        "congestion_sensitivity": _draw_from_interval(congestion_sensitivity_interval, r)
    }

# -----------------------------
# Topology generator
# -----------------------------
def generate_topology(
        *,
        num_fogs: int,
        num_edges: int,
        terminals_per_edge_interval: Tuple[int, int],
        cloud_storage_interval: Tuple[int, int],
        fog_storage_interval: Tuple[int, int],
        edge_storage_interval: Tuple[int, int],
        bandwidth_cloud_fog_interval: Tuple[float, float],
        bandwidth_fog_edge_interval: Tuple[float, float],
        bandwidth_edge_terminal_interval: Tuple[float, float],
        latency_cloud_fog_interval: Tuple[float, float],
        latency_fog_edge_interval: Tuple[float, float],
        latency_edge_terminal_interval: Tuple[float, float],
        congestion_sensitivity_cloud_interval: Tuple[float, float],
        congestion_sensitivity_fog_interval: Tuple[float, float],
        congestion_sensitivity_edge_interval: Tuple[float, float],
        generator_apps: List[Dict],
        consumer_apps: List[Dict],
        random_state: Optional[int] = 42,
) -> Tuple[nx.Graph, Dict, ApplicationInstanceManager]:
    r = _rng(random_state)
    G = nx.Graph()
    next_id = 0
    instance_manager = ApplicationInstanceManager()

    # 1. Create cloud node
    cloud_id = next_id
    next_id += 1
    G.add_node(cloud_id, **make_storage_node_attrs("cloud", r, cloud_storage_interval))

    # 2. Create fog nodes and connect to cloud
    fog_nodes = []
    for _ in range(num_fogs):
        fog_id = next_id
        next_id += 1
        G.add_node(fog_id, **make_storage_node_attrs("fog", r, fog_storage_interval))
        G.add_edge(
            cloud_id, fog_id,
            **make_link_attrs(r, bandwidth_cloud_fog_interval, latency_cloud_fog_interval, congestion_sensitivity_cloud_interval)
        )
        fog_nodes.append(fog_id)

    # 3. Create edge nodes and distribute among fogs
    edge_nodes = []
    edges_per_fog = [num_edges // num_fogs] * num_fogs
    for i in range(num_edges % num_fogs):
        edges_per_fog[i] += 1

    for fog_id, num_edges_in_fog in zip(fog_nodes, edges_per_fog):
        for _ in range(num_edges_in_fog):
            edge_id = next_id
            next_id += 1
            G.add_node(edge_id, **make_storage_node_attrs("edge", r, edge_storage_interval))
            G.add_edge(
                fog_id, edge_id,
                **make_link_attrs(r, bandwidth_fog_edge_interval, latency_fog_edge_interval, congestion_sensitivity_fog_interval)
            )
            edge_nodes.append(edge_id)

    # 4. Create terminal nodes under each edge
    terminal_nodes = []
    for edge_id in edge_nodes:
        num_terminals = _draw_from_interval(terminals_per_edge_interval, r, integer=True)
        for _ in range(num_terminals):
            terminal_id = next_id
            next_id += 1
            G.add_node(terminal_id, **make_terminal_node_attrs())
            G.add_edge(
                edge_id, terminal_id,
                **make_link_attrs(r, bandwidth_edge_terminal_interval, latency_edge_terminal_interval, congestion_sensitivity_edge_interval)
            )
            terminal_nodes.append(terminal_id)

    # 5. Deploy application instances
    deployment_info = deploy_application_instances(
        G, generator_apps, consumer_apps, terminal_nodes, instance_manager, r
    )

    # 6. Connect generators to consumers
    for generator_instance_id in instance_manager.generator_instances:
        instance_manager.connect_generator_to_consumers(generator_instance_id, num_consumers=3,r=r)

    return G, deployment_info, instance_manager

def deploy_application_instances(
        G: nx.Graph,
        generator_apps: List[Dict],
        consumer_apps: List[Dict],
        terminal_nodes: List[int],
        instance_manager: ApplicationInstanceManager,
        r: random.Random
) -> Dict[str, List]:
    storage_nodes = [n for n, d in G.nodes(data=True) if d.get("type") in ["cloud", "fog", "edge"]]

    deployment_info = {
        "node_instances": {},
        "generator_apps": [],
        "consumer_apps": []
    }

    # Deploy generator app instances
    for app in generator_apps:
        app_copy = app.copy()

        num_terminal_instances = int(len(terminal_nodes) * app["terminals_deployment_share"])
        num_storage_instances = int(len(storage_nodes) * app["storage_nodes_deployment_share"])

        terminal_targets = r.sample(terminal_nodes, min(num_terminal_instances, len(terminal_nodes)))
        storage_targets = r.sample(storage_nodes, min(num_storage_instances, len(storage_nodes)))
        all_targets = terminal_targets + storage_targets

        for node_id in all_targets:
            instance = instance_manager.create_generator_instance(app_copy, node_id, r)
            G.nodes[node_id]["generator_instances"].append(instance["instance_id"])

            if node_id not in deployment_info["node_instances"]:
                deployment_info["node_instances"][node_id] = []
            deployment_info["node_instances"][node_id].append(instance["instance_id"])

        app_copy["deployed_instances"] = len(all_targets)
        app_copy["deployed_nodes"] = all_targets
        deployment_info["generator_apps"].append(app_copy)

    # Deploy consumer app instances
    for app in consumer_apps:
        app_copy = app.copy()

        num_terminal_instances = int(len(terminal_nodes) * app["terminals_deployment_share"])
        num_storage_instances = int(len(storage_nodes) * app["storage_nodes_deployment_share"])

        terminal_targets = r.sample(terminal_nodes, min(num_terminal_instances, len(terminal_nodes)))
        storage_targets = r.sample(storage_nodes, min(num_storage_instances, len(storage_nodes)))
        all_targets = terminal_targets + storage_targets

        for node_id in all_targets:
            instance = instance_manager.create_consumer_instance(app_copy, node_id, r)
            G.nodes[node_id]["consumer_instances"].append(instance["instance_id"])

            if node_id not in deployment_info["node_instances"]:
                deployment_info["node_instances"][node_id] = []
            deployment_info["node_instances"][node_id].append(instance["instance_id"])

        app_copy["deployed_instances"] = len(all_targets)
        app_copy["deployed_nodes"] = all_targets
        deployment_info["consumer_apps"].append(app_copy)

    return deployment_info

# -----------------------------
# Latency cache
# -----------------------------
class ShortestPathCache:
    def __init__(self, G: nx.Graph):
        self.G = G
        self.storage_nodes = [n for n, d in G.nodes(data=True)
                              if d.get("type") in {"cloud", "fog", "edge"}]
        self.terminal_info = self._extract_terminal_info()
        self.distance_matrix = self._precompute()

    def _extract_terminal_info(self) -> Dict[int, Tuple[int, float]]:
        info: Dict[int, Tuple[int, float]] = {}
        for n, d in self.G.nodes(data=True):
            if d.get("type") == "terminal":
                neigh = list(self.G.neighbors(n))
                if neigh:
                    edge_n = neigh[0]
                    lat = self.G[n][edge_n].get("latency", 0)
                    info[n] = (edge_n, lat)
        return info

    def _precompute(self) -> Dict[int, Dict[int, float]]:
        dist: Dict[int, Dict[int, float]] = {s: {} for s in self.storage_nodes}
        for s in self.storage_nodes:
            lengths = nx.single_source_dijkstra_path_length(self.G, source=s, weight="latency")
            for t in self.storage_nodes:
                if s == t:
                    dist[s][t] = 0.0
                else:
                    dist[s][t] = float(lengths.get(t, float("inf")))
        return dist

    def get_latency(self, u: int, v: int) -> float:
        if u == v:
            return 0.0

        is_term_u = u in self.terminal_info
        is_term_v = v in self.terminal_info

        if is_term_u and is_term_v:
            eu, au = self.terminal_info[u]
            ev, av = self.terminal_info[v]
            core = self.distance_matrix.get(eu, {}).get(ev, float("inf"))
            return float(au + core + av)

        if is_term_u:
            eu, au = self.terminal_info[u]
            core = self.distance_matrix.get(eu, {}).get(v, float("inf"))
            return float(au + core)

        if is_term_v:
            ev, av = self.terminal_info[v]
            core = self.distance_matrix.get(u, {}).get(ev, float("inf"))
            return float(core + av)

        return float(self.distance_matrix.get(u, {}).get(v, float("inf")))

# -----------------------------
# Visualization
# -----------------------------
def visualize_topology(
        G: nx.Graph,
        *,
        hide_terminals: bool = False,
        show_edge_latencies: bool = True,
        title: str = "Topology"
):
    H = G if not hide_terminals else G.subgraph(
        [n for n, d in G.nodes(data=True) if d.get("type") != "terminal"]
    ).copy()

    pos = nx.kamada_kawai_layout(H)

    color_by_type = {
        "cloud": "skyblue",
        "fog": "orange",
        "edge": "lightgreen",
        "terminal": "gray",
    }
    colors = [color_by_type.get(H.nodes[n].get("type"), "gray") for n in H.nodes]

    plt.figure(figsize=(12, 9))
    nx.draw(
        H, pos,
        with_labels=False,
        node_color=colors,
        node_size=800,
        edge_color="gray",
        linewidths=0.8,
        alpha=0.95
    )

    edge_labels = {}
    if show_edge_latencies:
        edge_labels = {(u, v): f"{H[u][v].get('latency', ''):.1f}" for u, v in H.edges}

    if edge_labels:
        nx.draw_networkx_edge_labels(H, pos, edge_labels=edge_labels, font_size=7, alpha=0.9)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=t,
                   markerfacecolor=c, markersize=10)
        for t, c in color_by_type.items()
    ]
    plt.legend(handles=handles, loc="upper left", frameon=False, title="Node type")

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# -----------------------------
# Usage example
# -----------------------------
if __name__ == "__main__":
    G, deployment_info, instance_manager = generate_topology(**args)
    cache = ShortestPathCache(G)

    print("=== TOPOLOGY DEPLOYMENT SUMMARY ===")
    print(f"Generated topology with {len(G.nodes)} nodes and {len(G.edges)} edges")

    storage_nodes = [n for n, d in G.nodes(data=True) if d.get("type") in ["cloud", "fog", "edge"]]
    terminal_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "terminal"]
    print(f"Storage nodes: {len(storage_nodes)} (cloud: 1, fog: {args['num_fogs']}, edge: {args['num_edges']})")
    print(f"Terminal nodes: {len(terminal_nodes)}")

    print(f"\n=== APPLICATION INSTANCE DEPLOYMENT ===")
    print(f"Total generator instances: {len(instance_manager.generator_instances)}")
    print(f"Total consumer instances: {len(instance_manager.consumer_instances)}")

    sample_generator = instance_manager.generator_instances[0]
    gen_instance = instance_manager.get_instance(sample_generator)
    print(f"\nSample Generator: {gen_instance['name']} serves {len(gen_instance['connected_consumers'])} consumers")

    sample_consumer = instance_manager.consumer_instances[0]
    cons_instance = instance_manager.get_instance(sample_consumer)
    print(f"Sample Consumer: {cons_instance['name']} reads from {len(cons_instance['connected_generators'])} generators")