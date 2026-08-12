import pandas as pd
import torch
from itertools import combinations


def build_edge_index(
    edge_csv,
    N=30
):
    """
    Build the graph used by the Graph Attention Network.

    Important representation:

        Original road network:
            road segments connect physical intersections.

        GAT graph:
            each road segment becomes a graph node.

        Two road-segment nodes are connected when
        their corresponding road segments share a
        physical endpoint.

    Parameters
    ----------
    edge_csv : str
        Path to the road-edge CSV file.

    N : int
        Number of road segments selected for the model.

    Returns
    -------
    selected_edges : list
        Edge IDs of the selected road segments.

    edge_id_map : dict
        Maps original edge IDs to model node indices.

    edge_index : torch.Tensor
        Graph connectivity in PyTorch Geometric format.

        Shape:
            (2, number_of_graph_connections)
    """

    # =========================================================
    # 1. LOAD ROAD NETWORK
    # =========================================================

    df = pd.read_csv(edge_csv)

    # The CSV contains the original road-network information.
    #
    # Important columns:
    #
    #     edge_id -> identifies a road segment
    #     u       -> starting physical node
    #     v       -> ending physical node
    #
    # A road segment therefore looks conceptually like:
    #
    #     u -------- v
    #          road
    #

    required_columns = {
        "edge_id",
        "u",
        "v"
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            + str(missing_columns)
        )


    # =========================================================
    # 2. SELECT THE ROAD SEGMENTS
    # =========================================================

    # The final model works with N selected road segments.
    #
    # These roads become the N graph nodes used by GAT.

    selected_edges = (
        df["edge_id"]
        .head(N)
        .tolist()
    )


    # Create a mapping from the original road ID
    # to the index used inside the neural network.
    #
    # Example:
    #
    #     Original edge ID 101 → model node 0
    #     Original edge ID 105 → model node 1
    #     Original edge ID 110 → model node 2

    edge_id_map = {
        edge_id: index
        for index, edge_id
        in enumerate(selected_edges)
    }


    # Only keep rows corresponding to the selected
    # road segments.

    selected_df = df[
        df["edge_id"].isin(
            selected_edges
        )
    ].copy()


    # =========================================================
    # 3. FIND WHICH ROADS SHARE INTERSECTIONS
    # =========================================================

    # node_to_edges stores:
    #
    #     physical intersection
    #              ↓
    #     roads touching that intersection
    #
    # Example:
    #
    #     Physical node 25
    #          ↓
    #     [Road A, Road B, Road C]
    #
    # This tells us that A, B and C are spatially connected.

    node_to_edges = {}


    for _, row in selected_df.iterrows():

        edge_id = row["edge_id"]

        u = row["u"]

        v = row["v"]


        # Add the road to the list of roads
        # touching physical node u.

        node_to_edges.setdefault(
            u,
            []
        ).append(
            edge_id
        )


        # Add the road to the list of roads
        # touching physical node v.

        node_to_edges.setdefault(
            v,
            []
        ).append(
            edge_id
        )


    # =========================================================
    # 4. CREATE ROAD-TO-ROAD CONNECTIONS
    # =========================================================

    # Each pair of roads sharing a physical endpoint
    # becomes an edge in the GAT graph.

    graph_connections = set()


    for roads in node_to_edges.values():

        # combinations(..., 2) gives every pair of
        # roads meeting at this physical node.

        for edge_a, edge_b in combinations(
            roads,
            2
        ):

            # Convert original road IDs into the
            # consecutive node indices used by GAT.

            a = edge_id_map[edge_a]

            b = edge_id_map[edge_b]


            # Add both directions.
            #
            #     A → B
            #     B → A
            #
            # This allows information to flow between
            # neighboring road segments in both directions.

            graph_connections.add(
                (a, b)
            )

            graph_connections.add(
                (b, a)
            )


    # =========================================================
    # 5. CONVERT GRAPH TO PYTORCH GEOMETRIC FORMAT
    # =========================================================

    # PyTorch Geometric expects edge_index in the form:
    #
    #     [source_nodes]
    #     [target_nodes]
    #
    # Example:
    #
    #     [[0, 1, 1, 2],
    #      [1, 0, 2, 1]]
    #
    # This represents:
    #
    #     0 → 1
    #     1 → 0
    #     1 → 2
    #     2 → 1

    if graph_connections:

        edge_index = torch.tensor(
            list(graph_connections),
            dtype=torch.long
        ).t().contiguous()

    else:

        # If no road segments share endpoints,
        # create an empty graph.

        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long
        )


    # =========================================================
    # 6. RETURN GRAPH INFORMATION
    # =========================================================

    return (
        selected_edges,
        edge_id_map,
        edge_index
    )
