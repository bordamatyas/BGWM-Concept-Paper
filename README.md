**Biologized Grid-space World Model (BGWM): Bottom-Up Cognitive Cascade and Allocentric Spatial Representation in Robotic Control 
Concept Draft**

Matyas Borda

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21311833.svg)](https://doi.org/10.5281/zenodo.21311833)

Abstract: 
The dominant paradigm of current Vision-Language-Action (VLA) robotic systems relies on monolithic, language-based architectures. This approach faces fundamental limitations: human language is a lossily compressed representation that flattens the spatiotemporal dimensions of physical world states, while the egocentric camera view might make planning unstable. In this work, we introduce a biologically inspired, modular world model architecture (BGWM - Biologized Grid-Space World Model) that aims to solve the problems of catastrophic forgetting and spatial hallucinations. Our system is built on three main pillars: (1) the structured decomposition of instructions and vision into Entity Aggregates, (2) an Allocentric Grid-Cell mechanism that creates a topological cognitive map independent of the robot’s movement, and (3) a Cognitive Cascade World Model, a bottom-up architecture operating with skip-connections and flexible bottlenecks. The system is trained based on a strictly staged curriculum following biological ontogeny, explicitly separating motor trajectory generation from linguistic reflection.

<img width="1537" height="1023" alt="fig1" src="https://github.com/user-attachments/assets/76b9c4ae-ab93-462e-adf5-2f554ec29f80" />

*Figure 1: Overview of the Input Stream and Allocentric Transformation. The system processes visual, semantic, and kinesthetic data (Egocentric Inputs), fusing them through the Grid-Cell Binder to create structured Allocentric Entity Aggregates.*

<img width="1844" height="853" alt="fig2" src="https://github.com/user-attachments/assets/d54f3cde-b3e5-4f4f-b9ce-2d705cfb4b1b" />

*Figure 2: Architecture of the Cognitive Cascade World Model. The bottom-up hierarchical structure utilizes flexible bottlenecks and skip-connections to process spatiotemporal data, ultimately feeding into the dedicated Action and Language heads.*


PyTorch Pseudocode (Grid-Cell Binder): The “Place Cell” block that projects objects onto the global spatial grid based on odometry using Cross-Attention.
import torch
import torch.nn as nn

class GridCellBinder(nn.Module):
    def __init__(self, num_grid_nodes=1024, d_model=512):
        super().__init__()
        # Learnable grid nodes of the global spatial grid (Queries)
        self.grid_nodes = nn.Parameter(torch.randn(1, num_grid_nodes, d_model))
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=8)

    def forward(self, entity_tokens, x_ego, T_odometry):
        # 1. Homogeneous transformation (X_allo = T_odo_inv * X_ego)
        T_inv = torch.linalg.inv(T_odometry)
        x_allo = torch.bmm(x_ego, T_inv.transpose(1, 2))
        
        # 2. Keys (Where is it?) and Values (What is it?)
        keys = self.allocentric_coord_proj(x_allo)
        values = entity_tokens 
        
        # 3. Topological Binding onto the spatial grid
        B = entity_tokens.size(0) # Batch size
        allocentric_world_state, _ = self.cross_attention(
            query=self.grid_nodes.repeat(B, 1, 1).transpose(0, 1), 
            key=keys.transpose(0, 1), 
            value=values.transpose(0, 1)
        )
        return allocentric_world_state.transpose(0, 1)

Read the full concept paper on Zenodo:  [https://zenodo.org/records/21311833](//zenodo.org/records/21311833)
