# How to use ```generate_data.py```

## Execution

```python
python3 generate_data.py
python3 generate_data.py --n-switches 2 --n-additions 2 --n-deletions 2
```

By default all parameters are set to 2. The above command will thus generate 6 graphs, unless : .

- the generated graph ends up with 0 edges (prevents garbage)

- the generated graph has > 50 or < 10 nodes (prevents getting away from the initial distribution)

---

## Structure of the generated graphs

For each of the three procedure, I drew at random an integer **M**  between 1 and 0.2 * (G.number_of_nodes())

With this script you can generate 3 types of graphs from a single train/val graph G.

- "Switches"  : Takes the adjacency matrix and picks **M** coordinates to flip (0s become 1s and conversely). Applies this symmetrically and ignores the diagonal.
- "Additions" : Inserts **M** nodes and connects them at random as so : for each new node, pick an existing node at random and get its degree. This degree is the number of connections assigned to the new node. This allows for new nodes to follow the same patterns as other nodes in G.
- "Deletions" : Randomly remove **M** nodes.

Of course some choices are arbitrary and other procedures could be done, feel free to discuss about it if you feel we can do smth else.

---

## Is it useful ?

Yes. With the default model, going from the original data to the original data + the generated data (6 synthetic graphs per original one) makes the MAE go from 0.89 to 0.85. It is small but noticeable.