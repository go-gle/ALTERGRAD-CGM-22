import os
import json
import networkx as nx
import pandas as pd

def load_graph_features(train_descr):
    """Load graph features from description files."""
    from extract_feats import extract_feats 
    
    features_list = []
    for txt_file in os.listdir(train_descr):
        index = txt_file[6:-4]
        features_list.append([int(index)] + extract_feats(os.path.join(train_descr, txt_file)))
    
    df_feats = pd.DataFrame(features_list, columns=['index', 'nodes', 'edges', 'degree', 'triangles', 'clust_coef', 'max_k_core', 'communities'])
    df_feats.set_index('index', inplace=True)
    df_feats.sort_index(inplace=True)

    print("Sample of loaded graph features:")
    print(df_feats.head())
    return df_feats

def load_edge_lists(graph_dir):
    """Load edge lists from .edgelist and .graphml files."""
    edge_lists = {}
    for filename in os.listdir(graph_dir):
        filepath = os.path.join(graph_dir, filename)
        graph_index = int(filename.split("_")[1].split(".")[0])
        
        if filename.endswith(".edgelist"):
            G = nx.read_edgelist(filepath, nodetype=int)
        elif filename.endswith(".graphml"):
            G = nx.read_graphml(filepath)
            G = nx.convert_node_labels_to_integers(G)
        else:
            continue  # Skip unsupported file types
        
        edge_lists[graph_index] = list(G.edges())

    sample_keys = list(edge_lists.keys())[:5] 
    print("Sample edge lists:")
    for key in sample_keys:
        print(f"Graph index {key}: {edge_lists[key][:5]}")
    return edge_lists

def generate_fine_tuning_data(df_feats, edge_lists):
    """Generate fine-tuning data based on features and edge lists."""
    fine_tuning_data = []
    for idx, row in df_feats.iterrows():
        prompt = (
            f"<human>: Generate a graph with {int(row['nodes'])} nodes, {int(row['edges'])} edges, "
            f"an average degree of {row['degree']:.4f}, {int(row['triangles'])} triangles, "
            f"a clustering coefficient of {row['clust_coef']:.4f}, a maximum k-core of {int(row['max_k_core'])}, "
            f"and {int(row['communities'])} communities. "
            f"Output the edge list only in the format [(source, target), ...]. Ensure the graph respects the specified number of nodes, edges, and constraints. "
            f"Do not provide explanations or additional text.\n<assistant>:"
        )
        output = edge_lists.get(idx, [])
        if output:  # Ensure we have an edge list for this index
            fine_tuning_data.append({"prompt": prompt, "output": output})
    print("Sample of fine-tuning data:\n")
    print(fine_tuning_data[:2]) 
    return fine_tuning_data

def save_to_json(data, filename):
    """Save fine-tuning data to a JSON file."""
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Fine-tuning data saved to {filename}!")

def main():
    train_descr = './data/train/description/'
    graph_dir = './data/train/graph'
    output_file = 'fine_tuning_data.json'
    
    print("Loading graph features...")
    df_feats = load_graph_features(train_descr)
    
    print("Loading edge lists...")
    edge_lists = load_edge_lists(graph_dir)
    
    print("Generating fine-tuning data...")
    fine_tuning_data = generate_fine_tuning_data(df_feats, edge_lists)
    
    print("Saving fine-tuning data...")
    save_to_json(fine_tuning_data, output_file)

if __name__ == "__main__":
    main()
