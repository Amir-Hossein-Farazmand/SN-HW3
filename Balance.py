import pandas as pd
import networkx as nx
from collections import deque
import matplotlib.pyplot as plt
import os
import random

def solve_sign_prediction(file_path):

    df = pd.read_csv(file_path)
             
    df = df.iloc[:, :3]
    df.columns = ['u', 'v', 'sign']
        
    df['sign'] = pd.to_numeric(df['sign'], errors='coerce')
    df = df.dropna(subset=['sign'])

    # Separate known and unknown edges
    # '0' indicates an unknown sign
    known_edges = df[df['sign'] != 0].copy()
    unknown_edges = df[df['sign'] == 0].copy()

    # Build the graph using known edges
    G = nx.Graph()
    for _, row in known_edges.iterrows():
        G.add_edge(row['u'], row['v'], sign=row['sign'])

    # Determine structural balance partitioning using BFS
    node_groups = {}
    
    # Handle potentially disconnected components in the known graph
    components = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    
    for component in components:
        start_node = list(component.nodes())[0]
        node_groups[start_node] = 0 
        
        queue = deque([start_node])
        
        while queue:
            curr = queue.popleft()
            current_group = node_groups[curr]
            
            for neighbor in component.neighbors(curr):
                if neighbor not in node_groups:
                    edge_sign = component[curr][neighbor]['sign']
                    
                    # If edge is +1, neighbor is in SAME group
                    # If edge is -1, neighbor is in OPPOSITE group
                    if edge_sign == 1:
                        node_groups[neighbor] = current_group
                    else:
                        node_groups[neighbor] = 1 - current_group
                    
                    queue.append(neighbor)

    # Predict signs for unknown edges
    predicted_signs = []
    for _, row in unknown_edges.iterrows():
        u, v = row['u'], row['v']
        
        if u in node_groups and v in node_groups:
            group_u = node_groups[u]
            group_v = node_groups[v]
            
            if group_u == group_v:
                predicted_signs.append(1)
            else:
                predicted_signs.append(-1)
        else:
            predicted_signs.append(0) 

    unknown_edges['sign'] = predicted_signs
    
    # Combine results
    full_df = pd.concat([known_edges, unknown_edges]).sort_index()
    
    return full_df, unknown_edges

def solve_balance_test(file_path):

    df = pd.read_csv(file_path)
             
    df = df.iloc[:, :3]
    df.columns = ['u', 'v', 'sign']
    df['sign'] = pd.to_numeric(df['sign'], errors='coerce')
    df = df.dropna(subset=['sign'])

    # Build full graph
    G = nx.Graph()
    G.add_edges_from([(row['u'], row['v'], {'sign': row['sign']}) for _, row in df.iterrows()])
        
    # Super-node Generation
    # Construct a graph with ONLY positive edges
    positive_edges = [(u, v) for u, v, d in G.edges(data=True) if d['sign'] == 1]
    G_pos = nx.Graph()
    G_pos.add_nodes_from(G.nodes()) 
    G_pos.add_edges_from(positive_edges)
        
    # Find connected components (Super-nodes)
    super_node_components = list(nx.connected_components(G_pos))
        
    # Map each node to a super-node ID
    node_to_supernode = {}
    for idx, component in enumerate(super_node_components):
        for node in component:
            node_to_supernode[node] = idx
        
    # Construct Reduced Graph & Detect Contradictions
    negative_edges = [(u, v) for u, v, d in G.edges(data=True) if d['sign'] == -1]
        
    # Check Type 1: Negative edge within the same super-node
    for u, v in negative_edges:
        if node_to_supernode[u] == node_to_supernode[v]:
            return {
                'is_balanced': False,
                'message': "Unbalanced: Negative edge found inside a super-node.",
                'super_nodes': node_to_supernode,
                'num_super_nodes': len(super_node_components)
            }

    # Check Type 2: Reduced graph must be bipartite
    # Construct reduced graph where nodes are super-node IDs
    G_reduced = nx.Graph()
    G_reduced.add_nodes_from(range(len(super_node_components)))
        
    for u, v in negative_edges:
        u_super = node_to_supernode[u]
        v_super = node_to_supernode[v]
        if u_super != v_super:
            G_reduced.add_edge(u_super, v_super)
        
    if not nx.is_bipartite(G_reduced):
        return {
            'is_balanced': False,
            'message': "Unbalanced: Reduced graph of super-nodes is not bipartite (contains odd cycle).",
            'super_nodes': node_to_supernode,
            'num_super_nodes': len(super_node_components)
        }
            
    return {
        'is_balanced': True,
        'message': "Balanced",
        'super_nodes': node_to_supernode,
        'num_super_nodes': len(super_node_components)
    }


def solve_clusterability(file_path):

    df = pd.read_csv(file_path)
        
    df = df.iloc[:, :3]
    df.columns = ['u', 'v', 'sign']
        
    # Create Full Graph
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(row['u'], row['v'], sign=row['sign'])
            
    # Method: Connected Components on Positive Edges
    # Create a subgraph with only positive edges
    positive_edges = [(u, v) for u, v, d in G.edges(data=True) if d['sign'] == 1]
    G_pos = nx.Graph()
    G_pos.add_nodes_from(G.nodes()) 
    G_pos.add_edges_from(positive_edges)
        
    # Find connected components 
    clusters = list(nx.connected_components(G_pos))
        
    # Prepare report data
    cluster_sizes = {}
    node_to_cluster = {}
        
    for idx, cluster in enumerate(clusters):
        cluster_sizes[idx] = len(cluster)
        for node in cluster:
            node_to_cluster[node] = idx
                
    # Prepare color list for visualization
    node_colors = [node_to_cluster[n] for n in G.nodes()]
        
    return {
        'graph': G,
        'clusters': clusters,
        'cluster_sizes': cluster_sizes,
        'node_to_cluster': node_to_cluster,
        'node_colors': node_colors
    }

def calculate_line_index_score(G, node_clusters, alpha=0.5):

    P = 0
    N = 0
    
    for u, v, data in G.edges(data=True):
        sign = data['sign']
        
        # Check if nodes are in the map (handle potential missing nodes safely)
        if u not in node_clusters or v not in node_clusters:
            continue
            
        c_u = node_clusters[u]
        c_v = node_clusters[v]
        
        if c_u == c_v:
            # Within cluster
            if sign == -1:
                N += 1
        else:
            # Between clusters
            if sign == 1:
                P += 1
                
    line_index = (alpha * P) + ((1 - alpha) * N)
    return line_index, P, N

def solve_line_index(file_path, alpha=0.5, num_clusters=4):

    df = pd.read_csv(file_path)
    df = df.iloc[:, :3]
    df.columns = ['u', 'v', 'sign']
        
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(row['u'], row['v'], sign=row['sign'])
            
    nodes = list(G.nodes())
        
    # Random Clustering
    random.seed(42) 
    random_clusters = {node: random.randint(0, num_clusters - 1) for node in nodes}
        
    initial_li, init_P, init_N = calculate_line_index_score(G, random_clusters, alpha)
        
    # Heuristic Clustering (Greedy Local Search)
    # Iteratively move nodes to the cluster that minimizes local cost
        
    current_clusters = random_clusters.copy()
    max_iter = 50
        
    for i in range(max_iter):
        improvement_made = False
        # Shuffle nodes to avoid processing order bias
        random.shuffle(nodes)
            
        for node in nodes:
            current_k = current_clusters[node]
            best_k = current_k
                
            # Calculate current contribution to cost (Line Index)
            # We only need to check edges connected to 'node'
            neighbors = G[node]
                
            best_local_cost = float('inf')
                
            # Try placing node in every possible cluster
            for k in range(num_clusters):
                local_P = 0
                local_N = 0
                    
                for neighbor, data in neighbors.items():
                    sign = data['sign']
                    neighbor_k = current_clusters[neighbor]
                        
                    if neighbor_k == k:
                        # If we put 'node' in cluster k, this edge is WITHIN cluster
                        # Bad if sign is -1
                        if sign == -1:
                            local_N += 1
                    else:
                        # Edge is BETWEEN clusters
                        # Bad if sign is 1
                        if sign == 1:
                            local_P += 1
                    
                # Local penalty score for this node assignment
                cost = alpha * local_P + (1 - alpha) * local_N
                    
                if cost < best_local_cost:
                    best_local_cost = cost
                    best_k = k
                
            # If we found a better cluster for this node, move it
            if best_k != current_k:
                current_clusters[node] = best_k
                improvement_made = True
            
        # If no nodes moved in a full pass, we have converged
        if not improvement_made:
            break
                
    final_li, final_P, final_N = calculate_line_index_score(G, current_clusters, alpha)
        
    return {
        'initial_random': {
            'LI': initial_li,
            'P': init_P,
            'N': init_N
        },
        'optimized': {
            'LI': final_li,
            'P': final_P,
            'N': final_N
        },
        'iterations': i + 1,
        'final_assignments': current_clusters
    }

def solve_transitivity(file_path):

    df = pd.read_csv(file_path) 
    df = df.iloc[:, :2]
    df.columns = ['source', 'target']
        
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row['source'], row['target'])
            
    initial_edge_count = G.number_of_edges()
        
    # Check Transitivity
    transitive_triples = 0
    potential_triples = 0
        
    for j in G.nodes():
        predecessors = list(G.predecessors(j))
        successors = list(G.successors(j))
            
        for i in predecessors:
            for k in successors:
                if i == k: continue
                    
                potential_triples += 1
                if G.has_edge(i, k):
                    transitive_triples += 1
                        
    ratio = 0
    if potential_triples > 0:
        ratio = transitive_triples / potential_triples
            
    # Transitive Closure
    G_closure = G.copy()
    added_edges = []
        
    while True:
        new_edges_this_round = set()
            
        # Efficiently find missing transitive edges
        for j in G_closure.nodes():
            preds = list(G_closure.predecessors(j))
            succs = list(G_closure.successors(j))
            for i in preds:
                for k in succs:
                    if i == k: continue
                    if not G_closure.has_edge(i, k):
                        new_edges_this_round.add((int(i), int(k)))
            
        if not new_edges_this_round:
            break
                
        for u, v in new_edges_this_round:
            G_closure.add_edge(u, v)
            added_edges.append((u, v))
        
    final_edge_count = G_closure.number_of_edges()
        
    return {
        'transitive_triples': transitive_triples,
        'potential_triples': potential_triples,
        'transitivity_ratio': ratio,
        'initial_edges': initial_edge_count,
        'added_edges': added_edges,
        'added_count': len(added_edges),
        'final_total_edges': final_edge_count
    }
