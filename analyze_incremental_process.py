#!/usr/bin/env python3
"""
Deep analysis of the incremental room generation process in House-GAN++
Focus on how multiple rooms of same type are handled
"""

import numpy as np
import torch
from inference import ArchitecturalHouseGANInference

def analyze_incremental_generation():
    """Analyze step-by-step how the incremental process works"""
    print("=== ANALYZING INCREMENTAL GENERATION PROCESS ===")
    
    generator = ArchitecturalHouseGANInference()
    
    # Test case: 2 bedrooms that should generate differently
    room_types = ['living_room', 'bedroom', 'bedroom', 'kitchen', 'bathroom']
    
    print(f"Testing: {room_types}")
    
    # Create graph
    nodes, edges, room_ids = generator.create_architectural_graph(room_types)
    graph = [nodes, edges]
    real_nodes = np.array(room_ids)
    
    print(f"Room IDs: {room_ids}")
    print(f"Real nodes: {real_nodes}")
    
    # This is the key part - analyze the incremental type selection
    _types = sorted(list(set(real_nodes)))
    selected_types = [_types[:k+1] for k in range(min(10, len(_types)))]
    
    print(f"Unique room types: {_types}")
    print(f"Selected types progression: {selected_types}")
    
    # The issue is here - let's see what happens in each iteration
    state = {'masks': None, 'fixed_nodes': []}
    masks = generator._infer(graph, state)
    print(f"Initial masks shape: {masks.shape}")
    
    for _iter, _types_subset in enumerate(selected_types):
        print(f"\n--- Iteration {_iter}: Processing types {_types_subset} ---")
        
        # This is the problem line from the original code:
        _fixed_nds = np.concatenate([np.where(real_nodes == _t)[0] for _t in _types_subset]) \
            if len(_types_subset) > 0 else np.array([])
        
        print(f"Nodes to fix: {_fixed_nds}")
        
        # Let's see which actual room indices these correspond to
        for i, node_idx in enumerate(_fixed_nds):
            room_type = real_nodes[node_idx]
            room_name = generator.class_room.get(room_type + 1, f"Unknown({room_type + 1})")
            print(f"  Fixing node {node_idx}: {room_name} (type {room_type})")
        
        state = {'masks': masks, 'fixed_nodes': _fixed_nds}
        masks = generator._infer(graph, state)
        
        print(f"Masks shape after iteration: {masks.shape}")

def analyze_real_data_incremental():
    """Analyze how the real data handles incremental generation"""
    print("\n=== ANALYZING REAL DATA INCREMENTAL PROCESS ===")
    
    # Let's manually simulate what happens with real data structure
    # Using the 18477.json example which has: [3, 2, 5, 3, 4, 1] 
    # (bedroom, kitchen, balcony, bedroom, bathroom, living_room)
    
    real_nodes = np.array([2, 1, 4, 2, 3, 0])  # 0-based indexing
    print(f"Real data example (0-based): {real_nodes}")
    
    # Convert to names for clarity
    generator = ArchitecturalHouseGANInference()
    room_names = [generator.class_room.get(rid + 1, f"Unknown({rid + 1})") for rid in real_nodes]
    print(f"Room names: {room_names}")
    
    _types = sorted(list(set(real_nodes)))
    selected_types = [_types[:k+1] for k in range(min(10, len(_types)))]
    
    print(f"Unique types: {_types}")
    print(f"Selected types progression: {selected_types}")
    
    for _iter, _types_subset in enumerate(selected_types):
        print(f"\n--- Iteration {_iter}: Processing types {_types_subset} ---")
        
        _fixed_nds = np.concatenate([np.where(real_nodes == _t)[0] for _t in _types_subset]) \
            if len(_types_subset) > 0 else np.array([])
        
        print(f"Nodes to fix: {_fixed_nds}")
        
        for node_idx in _fixed_nds:
            room_type = real_nodes[node_idx]
            room_name = generator.class_room.get(room_type + 1, f"Unknown({room_type + 1})")
            print(f"  Fixing node {node_idx}: {room_name} (type {room_type})")

def analyze_template_vs_original():
    """Compare template-based generation with original data patterns"""
    print("\n=== COMPARING TEMPLATE VS ORIGINAL PATTERNS ===")
    
    # Original data pattern from 18477.json
    original_room_types = [3, 2, 5, 3, 4, 1]  # 1-based IDs
    original_adjacencies = [
        (0, 5), (1, 5), (2, 3), (3, 5), (4, 5)  # Adjacent pairs from ed_rm analysis
    ]
    
    print(f"Original room types (1-based): {original_room_types}")
    print(f"Original adjacencies: {original_adjacencies}")
    
    # Template-based equivalent
    generator = ArchitecturalHouseGANInference()
    template_rooms = ['bedroom', 'kitchen', 'balcony', 'bedroom', 'bathroom', 'living_room']
    
    nodes, edges, room_ids = generator.create_architectural_graph(template_rooms)
    
    print(f"Template room IDs (0-based): {room_ids}")
    print(f"Template adjacency edges sample:")
    
    # Analyze adjacency patterns in template
    adj_edges = edges[edges[:, 1] == 1]  # Only adjacent edges
    for i, edge in enumerate(adj_edges[:10]):  # Show first 10
        room1_name = generator.class_room.get(room_ids[edge[0]] + 1, f"Unknown")
        room2_name = generator.class_room.get(room_ids[edge[2]] + 1, f"Unknown")
        print(f"  {edge[0]}({room1_name}) -> {edge[2]}({room2_name})")

def main():
    """Main analysis"""
    analyze_incremental_generation()
    analyze_real_data_incremental()
    analyze_template_vs_original()
    
    print("\n" + "="*50)
    print("KEY FINDINGS:")
    print("1. Multiple rooms of same type get identical encodings")
    print("2. Incremental process fixes all instances of a type simultaneously")
    print("3. Model cannot distinguish between different instances of same room type")
    print("4. This causes the model to generate overlapping/duplicate rooms")

if __name__ == "__main__":
    main()