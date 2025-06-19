#!/usr/bin/env python3
"""
Debug script to analyze room generation issues in House-GAN++
Compares original dataset patterns vs synthetic data generation
"""

import numpy as np
import json
import torch
from collections import Counter, defaultdict
from inference import ArchitecturalHouseGANInference
import glob

def analyze_original_data():
    """Analyze patterns in original dataset JSON files"""
    print("=== ANALYZING ORIGINAL DATASET ===")
    
    json_files = glob.glob('./data/json/*.json')
    room_type_counts = Counter()
    room_distributions = []
    adjacency_patterns = defaultdict(list)
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        room_types = data['room_type']
        ed_rm = data['ed_rm']
        
        # Count room types (excluding doors and exterior walls)
        functional_rooms = [rt for rt in room_types if rt not in [15, 17]]
        room_type_counts.update(functional_rooms)
        room_distributions.append(functional_rooms)
        
        # Analyze adjacency patterns
        for i, room1 in enumerate(functional_rooms):
            for j, room2 in enumerate(functional_rooms):
                if i < j:  # Avoid duplicates
                    # Check if rooms are adjacent by looking at edge-room mappings
                    is_adjacent = any(
                        i in edge_rooms and j in edge_rooms 
                        for edge_rooms in ed_rm
                    )
                    adjacency_patterns[(room1, room2)].append(is_adjacent)
    
    print(f"Total samples: {len(json_files)}")
    print(f"Room type distribution (functional rooms only):")
    
    # Convert room IDs to names for readability
    ROOM_CLASS = {"living_room": 1, "kitchen": 2, "bedroom": 3, "bathroom": 4, 
                  "balcony": 5, "entrance": 6, "dining room": 7, "study room": 8,
                  "storage": 10}
    CLASS_ROM = {v: k for k, v in ROOM_CLASS.items()}
    
    for room_id, count in sorted(room_type_counts.items()):
        room_name = CLASS_ROM.get(room_id, f"Unknown({room_id})")
        print(f"  {room_name}: {count} instances")
    
    # Analyze room counts per floorplan
    room_counts_per_plan = [len(rooms) for rooms in room_distributions]
    print(f"Rooms per floorplan - Min: {min(room_counts_per_plan)}, Max: {max(room_counts_per_plan)}, Avg: {np.mean(room_counts_per_plan):.1f}")
    
    # Count multiple instances of same room type
    multiple_room_stats = defaultdict(int)
    for rooms in room_distributions:
        room_counter = Counter(rooms)
        for room_type, count in room_counter.items():
            if count > 1:
                room_name = CLASS_ROM.get(room_type, f"Unknown({room_type})")
                multiple_room_stats[room_name] += 1
    
    print(f"Floorplans with multiple instances of same room type:")
    for room_name, count in multiple_room_stats.items():
        print(f"  {room_name}: {count} floorplans have multiple instances")
    
    return room_distributions, adjacency_patterns

def analyze_synthetic_generation():
    """Analyze how synthetic data generation differs from original"""
    print("\n=== ANALYZING SYNTHETIC GENERATION ===")
    
    generator = ArchitecturalHouseGANInference()
    
    # Test different configurations
    test_configs = [
        (['living_room', 'kitchen', 'bathroom'], "Studio"),
        (['living_room', 'bedroom', 'kitchen', 'bathroom'], "1BR"),
        (['living_room', 'bedroom', 'bedroom', 'kitchen', 'bathroom'], "2BR"),
        (['living_room', 'bedroom', 'bedroom', 'bedroom', 'kitchen', 'bathroom', 'dining room'], "3BR House"),
        (['living_room', 'kitchen', 'balcony'], "Balcony Test"),
        (['living_room', 'study room'], "Study Room Test"),
    ]
    
    for i, (room_types, description) in enumerate(test_configs):
        print(f"\n--- {description} ---")
        print(f"Requested: {room_types}")
        
        try:
            # Create graph and analyze
            nodes, edges, room_ids = generator.create_architectural_graph(room_types)
            
            print(f"Generated node tensor shape: {nodes.shape}")
            print(f"Generated edges shape: {edges.shape if len(edges) > 0 else 'No edges'}")
            print(f"Room IDs mapping: {room_ids}")
            
            # Convert room_ids back to names for comparison
            generated_rooms = [generator.class_room.get(rid + 1, f"Unknown({rid + 1})") for rid in room_ids]
            print(f"Generated rooms: {generated_rooms}")
            
            # Check for issues
            room_counter = Counter(generated_rooms)
            multiple_rooms = [(room, count) for room, count in room_counter.items() if count > 1]
            if multiple_rooms:
                print(f"  WARNING: Multiple instances detected: {multiple_rooms}")
            
            # Analyze node encoding
            print(f"Node one-hot encoding check:")
            for j, room_id in enumerate(room_ids):
                room_name = generator.class_room.get(room_id + 1, f"Unknown({room_id + 1})")
                active_indices = torch.where(nodes[j] == 1.0)[0].tolist()
                print(f"  Room {j} ({room_name}): active at indices {active_indices}")
            
            # Test actual generation
            print("Running actual generation...")
            floorplan_image, masks, final_room_ids, real_nodes = generator.generate_floorplan(room_types)
            print(f"Final generation - Room IDs: {final_room_ids}")
            print(f"Real nodes array: {real_nodes}")
            
            final_rooms = [generator.class_room.get(rid + 1, f"Unknown({rid + 1})") for rid in final_room_ids]
            print(f"Final rooms: {final_rooms}")
            
            # Save debug image
            floorplan_image.save(f'./debug_floorplan_{i+1}_{description.replace(" ", "_")}.png')
            
        except Exception as e:
            print(f"Error in generation: {e}")
            import traceback
            traceback.print_exc()

def analyze_room_id_encoding():
    """Debug room ID encoding issues"""
    print("\n=== ANALYZING ROOM ID ENCODING ===")
    
    ROOM_CLASS = {"living_room": 1, "kitchen": 2, "bedroom": 3, "bathroom": 4, 
                  "balcony": 5, "entrance": 6, "dining room": 7, "study room": 8,
                  "storage": 10, "front door": 15, "unknown": 16, "interior_door": 17}
    CLASS_ROM = {v: k for k, v in ROOM_CLASS.items()}
    
    print("Room class mappings (ID -> Name):")
    for room_id, room_name in sorted(CLASS_ROM.items()):
        print(f"  {room_id}: {room_name}")
    
    # Test encoding with multiple bedrooms
    test_rooms = ['living_room', 'bedroom', 'bedroom', 'kitchen', 'bathroom']
    print(f"\nTesting encoding for: {test_rooms}")
    
    generator = ArchitecturalHouseGANInference()
    nodes, edges, room_ids = generator.create_architectural_graph(test_rooms)
    
    print(f"Room IDs: {room_ids}")
    print(f"Node tensor shape: {nodes.shape}")
    
    for i, (room_name, room_id) in enumerate(zip(test_rooms, room_ids)):
        expected_index = ROOM_CLASS[room_name] - 1  # 0-based
        actual_active = torch.where(nodes[i] == 1.0)[0].item()
        print(f"Room {i}: {room_name}")
        print(f"  Expected index: {expected_index}")
        print(f"  Actual index: {actual_active}")
        print(f"  Match: {'✓' if expected_index == actual_active else '✗'}")

def compare_adjacency_logic():
    """Compare adjacency rules between original and synthetic"""
    print("\n=== COMPARING ADJACENCY LOGIC ===")
    
    # Load sample original data
    with open('./data/json/18477.json', 'r') as f:
        original_data = json.load(f)
    
    room_types = original_data['room_type']
    ed_rm = original_data['ed_rm']
    
    # Filter functional rooms
    functional_indices = []
    functional_rooms = []
    for i, rt in enumerate(room_types):
        if rt not in [15, 17]:  # Not door or exterior wall
            functional_indices.append(i)
            functional_rooms.append(rt)
    
    print(f"Original sample functional rooms: {functional_rooms}")
    
    # Build original adjacency matrix
    n_rooms = len(functional_rooms)
    orig_adj_matrix = np.zeros((n_rooms, n_rooms), dtype=int)
    
    for i in range(n_rooms):
        for j in range(n_rooms):
            if i != j:
                orig_i = functional_indices[i]
                orig_j = functional_indices[j]
                # Check if rooms share an edge
                is_adjacent = any(
                    orig_i in edge_rooms and orig_j in edge_rooms 
                    for edge_rooms in ed_rm
                )
                orig_adj_matrix[i][j] = 1 if is_adjacent else -1
    
    print("Original adjacency matrix:")
    print(orig_adj_matrix)
    
    # Create synthetic equivalent
    CLASS_ROM = {1: "living_room", 2: "kitchen", 3: "bedroom", 4: "bathroom", 5: "balcony"}
    synthetic_rooms = [CLASS_ROM.get(rt, f"unknown_{rt}") for rt in functional_rooms]
    print(f"Synthetic equivalent: {synthetic_rooms}")
    
    generator = ArchitecturalHouseGANInference()
    try:
        nodes, edges, room_ids = generator.create_architectural_graph(synthetic_rooms)
        print(f"Synthetic edges shape: {edges.shape}")
        print("Sample synthetic edges:")
        if len(edges) > 0:
            for edge in edges[:10]:  # Show first 10 edges
                print(f"  {edge}")
    except Exception as e:
        print(f"Error in synthetic generation: {e}")

def main():
    """Main analysis function"""
    print("HOUSE-GAN++ ROOM GENERATION DEBUG ANALYSIS")
    print("=" * 50)
    
    # Analyze original dataset
    original_distributions, adjacency_patterns = analyze_original_data()
    
    # Analyze synthetic generation
    analyze_synthetic_generation()
    
    # Debug room ID encoding
    analyze_room_id_encoding()
    
    # Compare adjacency logic
    compare_adjacency_logic()
    
    print("\n" + "=" * 50)
    print("ANALYSIS COMPLETE")

if __name__ == "__main__":
    main()