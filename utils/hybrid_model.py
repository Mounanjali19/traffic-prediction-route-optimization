import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class HybridGAT_LSTM(nn.Module):
    """
    Hybrid Graph Attention Network + LSTM model.

    Purpose:
        Predict the future traffic speed of multiple road segments.

    Input:
        x_seq:
            Shape = (Batch, Time, Roads, Features)

            Batch    -> number of samples
            Time     -> 12 historical 5-minute observations
            Roads    -> selected road segments
            Features -> 8 traffic features

    Example:
        (1, 12, 30, 8)

    Output:
        Shape = (Batch, Roads)

        Each value represents the predicted traffic speed
        for one road segment.
    """

    def __init__(
        self,
        in_dim=8,
        gat_hidden=64,
        gat_heads=4,
        lstm_hidden=128,
        fusion_hidden=128
    ):
        super().__init__()

        # =====================================================
        # 1. GRAPH ATTENTION NETWORK
        # =====================================================
        #
        # GAT learns SPATIAL relationships between road segments.
        #
        # Each road segment is represented as a graph node.
        # Connected road segments can exchange information.
        #
        # Input:
        #     8 traffic features
        #
        # Output:
        #     64-dimensional spatial representation
        #
        # Four attention heads are used to allow the model
        # to learn different patterns of neighboring-road influence.
        #
        # concat=False means the outputs of the attention heads
        # are averaged rather than concatenated, so the output
        # remains 64-dimensional.
        #

        self.gat1 = GATConv(
            in_dim,
            gat_hidden,
            heads=gat_heads,
            concat=False
        )

        self.gat2 = GATConv(
            gat_hidden,
            gat_hidden,
            heads=gat_heads,
            concat=False
        )


        # =====================================================
        # 2. LSTM
        # =====================================================
        #
        # After GAT extracts spatial information from every
        # time step, the LSTM processes that information as a
        # temporal sequence.
        #
        # Input:
        #     64-dimensional GAT representation
        #
        # Output:
        #     128-dimensional temporal representation
        #
        # batch_first=True means the input format is:
        #
        #     (Batch, Time, Features)
        #

        self.lstm = nn.LSTM(
            input_size=gat_hidden,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True
        )


        # =====================================================
        # 3. FUSION LAYER
        # =====================================================
        #
        # We combine:
        #
        #     LSTM representation = 128 dimensions
        #     GAT representation  = 64 dimensions
        #
        # Total:
        #
        #     128 + 64 = 192
        #
        # This allows the final prediction to use both:
        #
        #     temporal information
        #     +
        #     spatial information
        #

        self.fusion = nn.Linear(
            lstm_hidden + gat_hidden,
            fusion_hidden
        )


        # =====================================================
        # 4. OUTPUT LAYER
        # =====================================================
        #
        # Produces one predicted speed for each road segment.
        #
        # Input:
        #     128-dimensional fused representation
        #
        # Output:
        #     1 predicted speed
        #

        self.out = nn.Linear(
            fusion_hidden,
            1
        )


    def forward(
        self,
        x_seq,
        edge_index
    ):
        """
        Forward pass through the Hybrid GAT-LSTM model.

        Parameters
        ----------
        x_seq : torch.Tensor
            Traffic history.

            Shape:
                (B, T, N, F)

            B = batch size
            T = number of historical time steps
            N = number of road segments
            F = number of traffic features

        edge_index : torch.Tensor
            Graph connectivity used by the GAT layers.

            Shape:
                (2, number_of_graph_edges)

        Returns
        -------
        torch.Tensor

            Predicted speed for every road segment.

            Shape:
                (B, N)
        """

        # -----------------------------------------------------
        # Extract dimensions from the input
        # -----------------------------------------------------

        B, T, N, F = x_seq.shape

        predictions = []


        # =====================================================
        # PROCESS EACH SAMPLE IN THE BATCH
        # =====================================================

        for b in range(B):

            # Stores the spatial representation produced
            # by GAT at every time step.

            gat_features_over_time = []


            # =================================================
            # PROCESS EACH TIME STEP WITH GAT
            # =================================================

            for t in range(T):

                # Features of every road at this time step.
                #
                # Shape:
                #
                #     (N, F)
                #
                # Example:
                #
                #     (30, 8)

                x_t = x_seq[b, t]


                # -------------------------------------------------
                # First GAT layer
                # -------------------------------------------------
                #
                # The graph structure tells GAT which roads
                # are connected.
                #
                # GAT aggregates information from neighboring
                # road segments while learning attention weights.
                #

                h1 = self.gat1(
                    x_t,
                    edge_index
                )

                h1 = torch.relu(h1)


                # -------------------------------------------------
                # Second GAT layer
                # -------------------------------------------------
                #
                # Refines the spatial representation after the
                # first neighborhood aggregation.
                #

                h2 = self.gat2(
                    h1,
                    edge_index
                )

                h2 = torch.relu(h2)


                # h2 shape:
                #
                #     (N, 64)
                #
                # Add a time dimension so that representations
                # from different time steps can later be stacked.

                gat_features_over_time.append(
                    h2.unsqueeze(0)
                )


            # =================================================
            # CREATE TEMPORAL SEQUENCE
            # =================================================

            # Concatenate all time steps.
            #
            # Before permutation:
            #
            #     (T, N, 64)

            gat_sequence = torch.cat(
                gat_features_over_time,
                dim=0
            )


            # -------------------------------------------------
            # Rearrange dimensions for LSTM
            # -------------------------------------------------
            #
            # We want every road to have its own time sequence:
            #
            #     Road 1: t1 → t2 → ... → t12
            #     Road 2: t1 → t2 → ... → t12
            #     ...
            #
            # Therefore:
            #
            #     (T, N, 64)
            #
            # becomes:
            #
            #     (N, T, 64)

            gat_sequence = gat_sequence.permute(
                1,
                0,
                2
            )


            # =================================================
            # LSTM TEMPORAL MODELING
            # =================================================
            #
            # The LSTM now learns how the spatial representation
            # of each road changes across the 12 historical
            # time steps.

            lstm_output, _ = self.lstm(
                gat_sequence
            )


            # -------------------------------------------------
            # Take the final time step
            # -------------------------------------------------
            #
            # Shape:
            #
            #     (N, 128)
            #
            # This represents the learned temporal state of
            # each road after processing the complete history.

            lstm_last = lstm_output[:, -1, :]


            # =================================================
            # LAST SPATIAL REPRESENTATION
            # =================================================
            #
            # We also retain the GAT representation from the
            # most recent time step.
            #
            # Shape:
            #
            #     (N, 64)

            gat_last = gat_sequence[:, -1, :]


            # =================================================
            # SPATIAL + TEMPORAL FUSION
            # =================================================
            #
            # Combine:
            #
            #     LSTM = 128
            #     GAT  = 64
            #
            # Result:
            #
            #     192 dimensions

            fused = torch.cat(
                [
                    lstm_last,
                    gat_last
                ],
                dim=1
            )


            # Reduce the combined representation:
            #
            #     192 → 128

            fused = self.fusion(
                fused
            )

            fused = torch.relu(
                fused
            )


            # =================================================
            # FINAL SPEED PREDICTION
            # =================================================
            #
            # One output for every road segment.
            #
            # Shape before squeeze:
            #
            #     (N, 1)
            #
            # Shape after squeeze:
            #
            #     (N,)

            prediction = self.out(
                fused
            ).squeeze(-1)


            # Add batch dimension back.

            predictions.append(
                prediction.unsqueeze(0)
            )


        # =====================================================
        # RETURN COMPLETE BATCH
        # =====================================================
        #
        # Final shape:
        #
        #     (B, N)
        #
        # Example:
        #
        #     (1, 30)
        #
        # meaning 30 predicted road speeds.

        return torch.cat(
            predictions,
            dim=0
        )
