import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
import numpy as np

class SchoolNetworkAnalyzer:
    def __init__(self, base_path):

        self.base_path = base_path
        self.days = [1, 30, 60, 90]
        self.graphs = {}     
        self.attributes = {}  
        
    def load_data(self):

        for day in self.days:
            # Load Properties
            prop_file = os.path.join(self.base_path, f'properties_day_{day}.csv')
            if not os.path.exists(prop_file):
                raise FileNotFoundError(f"File not found: {prop_file}")
            
            prop_df = pd.read_csv(prop_file)
            self.attributes[day] = prop_df.set_index('id')
            
            # Load Connections 
            conn_file = os.path.join(self.base_path, f'connections_day_{day}.csv')
            if not os.path.exists(conn_file):
                raise FileNotFoundError(f"File not found: {conn_file}")
            
            # Create Graph
            G = nx.Graph()
            
            # Add nodes with attributes
            for node_id, row in self.attributes[day].iterrows():
                G.add_node(node_id, **row.to_dict())
                
            # Add edges
            
                edges_df = pd.read_csv(conn_file, header=None)
                if isinstance(edges_df.iloc[0, 0], str):
                    edges_df = pd.read_csv(conn_file)
                else:
                    edges_df.columns = ['u', 'v']


            # Clean edges
            edge_list = []
            for _, row in edges_df.iterrows():
                try:
                    u, v = int(row.iloc[0]), int(row.iloc[1])
                    edge_list.append((u, v))
                except:
                    continue
            
            G.add_edges_from(edge_list)
            self.graphs[day] = G
            
        print("Data loaded successfully for days:", self.days)

    def analyze_closure_mechanisms(self):

        intervals = [(1, 30), (30, 60), (60, 90)]
        results = {}
        
        for t_prev, t_curr in intervals:
            G_prev = self.graphs[t_prev]
            G_curr = self.graphs[t_curr]
            
            edges_prev = set([frozenset(e) for e in G_prev.edges()])
            edges_curr = set([frozenset(e) for e in G_curr.edges()])
            
            # Identify newly formed edges
            new_edges = [tuple(e) for e in (edges_curr - edges_prev)]
            
            stats = {
                'new_edges_count': len(new_edges),
                'triadic_closure_count': 0,
                'membership_closure_count': 0, 
                'focal_smoking_closure_count': 0, 
                'pure_random_or_other': 0
            }
            
            for u, v in new_edges:
                if u not in G_prev.nodes or v not in G_prev.nodes:
                    continue 
                
                # Triadic Closure Check
                common_neighbors = list(nx.common_neighbors(G_prev, u, v))
                is_triadic = len(common_neighbors) > 0
                
                # Membership Closure
                node_u = G_prev.nodes[u]
                node_v = G_prev.nodes[v]
                
                c_u = node_u.get('class_number')
                c_v = node_v.get('class_number')
                
                is_membership = (c_u is not None) and (c_v is not None) and (c_u == c_v)
                
                # Focal Closure 
                smoke_u = node_u.get('smokes', 0)
                smoke_v = node_v.get('smokes', 0)
                is_focal_smoke = (smoke_u == 1 and smoke_v == 1)
                
                # Increment counters
                if is_triadic: stats['triadic_closure_count'] += 1
                if is_membership: stats['membership_closure_count'] += 1
                if is_focal_smoke: stats['focal_smoking_closure_count'] += 1
                
                if not (is_triadic or is_membership or is_focal_smoke):
                    stats['pure_random_or_other'] += 1
            
            results[f"{t_prev}->{t_curr}"] = stats
            
        return results

    def analyze_new_smokers(self):

        intervals = [(1, 30), (30, 60), (60, 90)]
        evolution_data = []
        
        for t_prev, t_curr in intervals:
            df_prev = self.attributes[t_prev]
            df_curr = self.attributes[t_curr]
            G_prev = self.graphs[t_prev]
            
            # Find students who switched smokes 0 -> 1
            new_smokers_ids = []
            for uid in df_curr.index:
                if df_prev.loc[uid, 'smokes'] == 0 and df_curr.loc[uid, 'smokes'] == 1:
                    new_smokers_ids.append(uid)
            
            for uid in new_smokers_ids:
                # Analyze peer group at t_prev
                friends = list(G_prev.neighbors(uid))
                num_friends = len(friends)
                if num_friends > 0:
                    smoker_friends = [f for f in friends if G_prev.nodes[f].get('smokes') == 1]
                    percent_smoker_friends = (len(smoker_friends) / num_friends) * 100
                else:
                    percent_smoker_friends = 0
                
                evolution_data.append({
                    'interval': f"{t_prev}->{t_curr}",
                    'student_id': uid,
                    'total_friends_prev': num_friends,
                    'smoker_friends_prev': len(smoker_friends),
                    'smoker_peer_pressure_pct': percent_smoker_friends
                })
                
        return pd.DataFrame(evolution_data)

    def compare_groups(self, day=90):

        df = self.attributes[day].reset_index()
        
        # Add degree centrality
        G = self.graphs[day]
        degrees = dict(G.degree())
        
        if 'id' in df.columns:
            df['degree'] = df['id'].map(degrees)
        else:
            df['degree'] = df.index.map(degrees)
        
        # Exact column mapping for aggregation
        agg_dict = {
            'studies': 'mean',
            'plays_football': 'mean',
            'watches_movies': 'mean',
            'club': 'mean',
            'degree': 'mean',
            'id': 'count'
        }
        
        existing_agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}

        # Group by smoking status
        comparison = df.groupby('smokes').agg(existing_agg_dict)
        
        # Rename 'id' to 'count' for clarity
        if 'id' in comparison.columns:
            comparison = comparison.rename(columns={'id': 'count'})
        
        return comparison

    def get_top_central_nodes(self, day=90, top_n=5):

        G = self.graphs[day]
        centrality = nx.degree_centrality(G)
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        # Enrich with attributes
        results = []
        for uid, score in sorted_nodes:
            attrs = G.nodes[uid]
            results.append({
                'id': uid,
                'degree_centrality': score,
                'raw_degree': G.degree(uid),
                'smokes': attrs.get('smokes'),
                'gender': attrs.get('gender'),
                'class': attrs.get('class_number')
            })
            
        return pd.DataFrame(results)

    def visualize_network_with_smokers(self, day=90):

        G = self.graphs[day]
        
        plt.figure(figsize=(14, 12)) 
        
        # Color mapping
        node_colors = []
        for node in G.nodes():
            if G.nodes[node].get('smokes') == 1:
                node_colors.append('red')
            else:
                node_colors.append('skyblue')
        
        pos = nx.spring_layout(G, k=0.6, iterations=100, seed=42)
        
        # Draw with adjusted visual parameters
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=60, alpha=0.9) 
        nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='gray', width=0.8) 
        
        plt.title(f"School Network Day {day} (Red=Smoker, Blue=Non-Smoker)")
        plt.axis('off')
        plt.show()