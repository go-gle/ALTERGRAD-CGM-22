import os
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import networkx as nx
from typing import List, Tuple, Dict
import json

from extract_feats import extract_numbers


class ConcatenatedFileDataset(Dataset):
    def __init__(self, consolidated_file: str):
        self.data: List[Dict[str, str]] = []
        # Read and parse each line as JSON
        with open(consolidated_file, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                # Store the entire dictionary - more flexible
                self.data.append(item)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.data[idx]

    def map(self, function, num_workers: int = None) -> 'ConcatenatedFileDataset':
        """Apply a function to all items in the dataset.
        
        Args:
            function (Callable): The function to apply to each item
            num_workers (int, optional): Number of workers for parallel processing
        
        Returns:
            ConcatenatedFileDataset: A new dataset with transformed items
        """
        # Create a new dataset instance
        new_dataset = ConcatenatedFileDataset.__new__(ConcatenatedFileDataset)
        
        if num_workers and num_workers > 0:
            # Parallel processing using Pool
            with Pool(num_workers) as p:
                new_dataset.data = list(p.map(function, self.data))
        else:
            # Sequential processing
            new_dataset.data = [function(item) for item in self.data]
            
        return new_dataset
    
if __name__ == '__main__':

    data_path = 'data/'

    for set_ in ['train', 'valid']:
        path = os.path.join(data_path, set_)
        graphs_path = os.path.join(path, 'graph/')
        prompts_path = os.path.join(path, 'description/')
        graphs = os.listdir(graphs_path)
        print(f'generating the jsonl for {path} set...')
        for graph in tqdm(graphs):
            #manage graphs
            graph_file = os.path.join(graphs_path, graph)
            if graph.endswith(".graphml"):
                G = nx.read_graphml(graph_file)
                G = nx.convert_node_labels_to_integers(G, ordering="sorted")
            else:
                G = nx.read_edgelist(graph_file)    

            #our answer
            edges_str = str(list(G.edges()))[1:-1].replace("'", "") 

            #manage prompts
            fname = graph.split('.')[0]
            txt_file = os.path.join(prompts_path, fname) + '.txt'
            with open(txt_file, 'r') as f:
                text = f.read()
                features = extract_numbers(text)
            #our prompt
            prompt = f'Give the graph edgelist associated to the following features.-Number of nodes: {features[0]}-Number of edges: {features[1]}-Average degree: {features[2]}-Number of triangles: {features[3]}-Clustering coefficient: {features[4]}-Max k cores: {features[5]}-Number of communities: {features[6]}'

            #write everything
            formated_pair = '{"prompt": "' + prompt + '", "answer": "' + edges_str + '"}\n'
            with open(data_path + f'{set_}.jsonl', 'a') as f:
                f.write(formated_pair)

    test_path =  os.path.join(data_path, 'test/')
    with open(test_path + 'test.txt', 'r') as ftest:
        with open(data_path + 'test.jsonl', 'a') as new_f:
            for text in tqdm(ftest.readlines()):
                features = extract_numbers(text)
                prompt = f'Give the graph edgelist associated to the following features.-Number of nodes: {features[0]}-Number of edges: {features[1]}-Average degree: {features[2]}-Number of triangles: {features[3]}-Clustering coefficient: {features[4]}-Max k cores: {features[5]}-Number of communities: {features[6]}'
                formated = '{"prompt": "' + prompt + '"}\n'
                new_f.write(formated)

    train = ConcatenatedFileDataset('data/train.jsonl')
    torch.save(train, 'data/train.pt')
    test = ConcatenatedFileDataset('data/test.jsonl')
    torch.save(train, 'data/test.pt')
    valid = ConcatenatedFileDataset('data/valid.jsonl')
    torch.save(train, 'data/valid.pt')