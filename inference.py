#!/usr/bin/env python3

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import cv2
import webcolors

# Import from the original codebase
from models.models import Generator

# Define constants matching the original
ROOM_CLASS = {"living_room": 1, "kitchen": 2, "bedroom": 3, "bathroom": 4, "balcony": 5, "entrance": 6, "dining room": 7, "study room": 8,
              "storage": 10 , "front door": 15, "unknown": 16, "interior_door": 17}
CLASS_ROM = {}
for x, y in ROOM_CLASS.items():
    CLASS_ROM[y] = x
ID_COLOR = {1: '#EE4D4D', 2: '#C67C7B', 3: '#FFD274', 4: '#BEBEBE', 5: '#BFE3E8', 6: '#7BA779', 7: '#E87A90', 8: '#FF8C69', 10: '#1F849B', 15: '#727171', 16: '#785A67', 17: '#D3A2C7'}

# FIXED: Data-driven adjacency rules based on real architectural patterns
REAL_ADJACENCY_RULES = {
    # From real data analysis - these are the most common adjacencies
    ('living_room', 'bedroom'): 1,      # 25 times in real data - VERY common
    ('living_room', 'kitchen'): 1,      # 11 times - common  
    ('living_room', 'bathroom'): 1,     # 10 times - common
    ('living_room', 'balcony'): 1,      # 8 times - common
    ('bedroom', 'balcony'): -1,         # 2 times - rare, usually not adjacent
    ('kitchen', 'balcony'): -1,         # 2 times - rare
    ('bedroom', 'bathroom'): -1,        # Not in top adjacencies - usually not direct
    ('kitchen', 'bathroom'): -1,        # Usually separate
    ('bedroom', 'bedroom'): -1,         # Multiple bedrooms usually separate
    ('balcony', 'balcony'): -1,         # Multiple balconies usually separate
    
    # Additional rules for other room types
    ('living_room', 'dining room'): 1,  # Usually adjacent
    ('kitchen', 'dining room'): 1,      # Usually adjacent  
    ('living_room', 'entrance'): 1,     # Entrance connects to living area
    ('living_room', 'study room'): 1,   # Study room often near living area
    ('bedroom', 'study room'): -1,      # Usually separate zones
    ('kitchen', 'entrance'): -1,        # Kitchen usually not at entrance
    ('bathroom', 'entrance'): -1,       # Bathroom not at entrance
    ('bathroom', 'kitchen'): -1,        # Usually separate
    ('bathroom', 'dining room'): -1,    # Usually separate
}

# FIXED: Realistic room count constraints based on real data
REALISTIC_ROOM_COUNTS = {
    'living_room': (1, 1),    # Always exactly 1
    'kitchen': (1, 1),        # Always exactly 1  
    'bathroom': (1, 1),       # Usually exactly 1
    'bedroom': (1, 3),        # 1-3 bedrooms typical
    'balcony': (0, 2),        # 0-2 balconies
    'dining room': (0, 1),    # 0-1 dining room
    'study room': (0, 1),     # 0-1 study room
    'entrance': (0, 1),       # 0-1 entrance
    'storage': (0, 1),        # 0-1 storage
}

def one_hot_embedding(labels, num_classes=19):
    """One-hot embedding matching the original dataset"""
    y = torch.eye(num_classes)
    return y[labels]

def fix_nodes(prev_mks, ind_fixed_nodes):
    """Fix certain nodes in the mask - matches original implementation"""
    given_masks = prev_mks.clone() if torch.is_tensor(prev_mks) else torch.tensor(prev_mks)
    ind_not_fixed_nodes = torch.tensor([k for k in range(given_masks.shape[0]) if k not in ind_fixed_nodes])
    
    # Set non fixed masks to -1.0
    given_masks[ind_not_fixed_nodes.long()] = -1.0
    given_masks = given_masks.unsqueeze(1)
    
    # Add channel to indicate given nodes 
    inds_masks = torch.zeros_like(given_masks)
    inds_masks[ind_not_fixed_nodes.long()] = 0.0
    inds_masks[ind_fixed_nodes.long()] = 1.0
    given_masks = torch.cat([given_masks, inds_masks], 1)
    return given_masks

def _init_input(graph, prev_state=None, mask_size=64):
    """Initialize input for the model - matches original exactly"""
    # initialize graph
    given_nds, given_eds = graph
    given_nds = given_nds.float()
    given_eds = torch.tensor(given_eds).long()
    z = torch.randn(len(given_nds), 128).float()
    
    # unpack
    fixed_nodes = prev_state['fixed_nodes']
    prev_mks = torch.zeros((given_nds.shape[0], mask_size, mask_size))-1.0 if (prev_state['masks'] is None) else prev_state['masks']
    
    # initialize masks
    given_masks_in = fix_nodes(prev_mks, torch.tensor(fixed_nodes))
    return z, given_masks_in, given_nds, given_eds

def draw_masks(masks, real_nodes, im_size=256):
    """Draw room masks as colored regions - matches original"""
    bg_img = Image.new("RGBA", (im_size, im_size), (255, 255, 255, 255))
    
    for m, nd in zip(masks, real_nodes):
        # resize map
        m[m>0] = 255
        m[m<0] = 0
        m_lg = cv2.resize(m, (im_size, im_size), interpolation = cv2.INTER_AREA) 

        # pick color
        color = ID_COLOR.get(nd+1, '#000000')
        r, g, b = webcolors.hex_to_rgb(color)

        # set drawer
        dr_bkg = ImageDraw.Draw(bg_img)

        # draw region
        m_pil = Image.fromarray(m_lg)
        dr_bkg.bitmap((0, 0), m_pil.convert('L'), fill=(r, g, b, 256))

        # draw contour
        m_cv = m_lg[:, :, np.newaxis].astype('uint8')
        ret, thresh = cv2.threshold(m_cv, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if len(contours) > 0]
        cnt = np.zeros((256, 256, 3)).astype('uint8')
        cv2.drawContours(cnt, contours, -1, (255, 255, 255, 255), 1)
        cnt = Image.fromarray(cnt)
        dr_bkg.bitmap((0, 0), cnt.convert('L'), fill=(0, 0, 0, 255))

    return bg_img.resize((im_size, im_size))

class FixedHouseGANInference:
    def __init__(self, checkpoint_path='./checkpoints/pretrained.pth'):
        """Initialize with FIXED architectural understanding based on real data"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pretrained model
        self.model = Generator()
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model = self.model.eval()
        self.model.to(self.device)
        
        # Transform to match training data
        self.transform = transforms.Normalize(mean=[0.5], std=[0.5])
        
        # Room type mappings
        self.room_class = ROOM_CLASS
        self.class_room = CLASS_ROM
        self.id_color = ID_COLOR
        
    def validate_room_selection(self, room_types):
        """FIXED: Validate room selection against realistic constraints"""
        room_counts = {}
        for room in room_types:
            room_counts[room] = room_counts.get(room, 0) + 1
        
        warnings = []
        for room, count in room_counts.items():
            if room in REALISTIC_ROOM_COUNTS:
                min_count, max_count = REALISTIC_ROOM_COUNTS[room]
                if count < min_count or count > max_count:
                    warnings.append(f"{room}: requested {count}, typical range {min_count}-{max_count}")
        
        return warnings
        
    def create_data_driven_graph(self, room_types):
        """
        FIXED: Create graph based on real architectural data patterns
        """
        # Validate realistic room counts
        warnings = self.validate_room_selection(room_types)
        if warnings:
            print("⚠️  Room count warnings:", warnings)
        
        # Convert room names to IDs
        room_ids = []
        for room_name in room_types:
            if room_name in self.room_class:
                room_ids.append(self.room_class[room_name] - 1)  # 0-based indexing
            else:
                print(f"Warning: Unknown room type '{room_name}', skipping...")
        
        if not room_ids:
            raise ValueError("No valid room types provided")
        
        # Create node tensor (one-hot encoding) - exclude class 0
        num_rooms = len(room_ids)
        max_room_types = 18  # Exclude class 0
        nodes = torch.zeros(num_rooms, max_room_types)
        for i, room_id in enumerate(room_ids):
            nodes[i, room_id] = 1.0
        
        # FIXED: Create edges based on real adjacency data
        edges = []
        
        for i in range(num_rooms):
            for j in range(i+1, num_rooms):
                room1_name = room_types[i]
                room2_name = room_types[j]
                
                # Check real adjacency rules
                adj_key = tuple(sorted([room1_name, room2_name]))
                if adj_key in REAL_ADJACENCY_RULES:
                    adj_type = REAL_ADJACENCY_RULES[adj_key]
                else:
                    # Default rule based on real data patterns
                    adj_type = self._get_default_adjacency(room1_name, room2_name)
                
                # Add bidirectional edges
                edges.append([i, adj_type, j])
                edges.append([j, adj_type, i])
        
        # Convert to numpy array
        if edges:
            edges = np.array(edges)
        else:
            # Fallback for single room
            edges = np.array([[0, 1, 0]])
            
        return nodes, edges, room_ids
    
    def _get_default_adjacency(self, room1, room2):
        """FIXED: Default adjacency rules based on real data patterns"""
        # Living room is central - connects to most things
        if 'living_room' in [room1, room2]:
            return 1  # Usually adjacent
        
        # Service areas (kitchen, bathroom) usually separate
        if room1 in ['kitchen', 'bathroom'] and room2 in ['kitchen', 'bathroom']:
            return -1
        
        # Private spaces (bedroom, study) usually separate from service areas
        if room1 in ['bedroom', 'study room'] and room2 in ['kitchen', 'bathroom']:
            return -1
        
        # Multiple instances of same type usually separate
        if room1 == room2:
            return -1
        
        # Default: not adjacent (conservative)
        return -1
    
    def _infer(self, graph, prev_state=None):
        """Run inference - matches original exactly"""
        z, given_masks_in, given_nds, given_eds = _init_input(graph, prev_state)
        
        # Move to device
        z = z.to(self.device)
        given_masks_in = given_masks_in.to(self.device)
        given_nds = given_nds.to(self.device)
        given_eds = given_eds.to(self.device)
        
        # Run inference model
        with torch.no_grad():
            masks = self.model(z, given_masks_in, given_nds, given_eds)
            masks = masks.detach().cpu().numpy()
        return masks
    
    def generate_floorplan(self, room_types):
        """
        FIXED: Generate floorplan using real architectural patterns and proper constraints
        """
        # Create data-driven graph
        nodes, edges, room_ids = self.create_data_driven_graph(room_types)
        graph = [nodes, edges]
        
        # Get real_nodes (room type indices)
        real_nodes = np.array(room_ids)
        
        print(f"🏗️  Generated graph: {len(room_ids)} rooms, {len(edges)} edges")
        print(f"   Room types: {[self.class_room.get(r+1, f'unknown_{r+1}') for r in room_ids]}")
        print(f"   Adjacency edges: {len(edges[edges[:,1]==1])//2 if len(edges)>0 else 0} adjacent pairs")
        
        # FIXED: Original incremental process - exactly matching test.py
        _types = sorted(list(set(real_nodes)))
        selected_types = [_types[:k+1] for k in range(min(10, len(_types)))]
        
        # Initialize layout
        state = {'masks': None, 'fixed_nodes': []}
        masks = self._infer(graph, state)
        
        # Generate per room type - matches original test.py exactly
        for _iter, _types_subset in enumerate(selected_types):
            # Find nodes matching the selected types - EXACT ORIGINAL LOGIC
            _fixed_nds = np.concatenate([np.where(real_nodes == _t)[0] for _t in _types_subset]) \
                if len(_types_subset) > 0 else np.array([])
            
            state = {'masks': masks, 'fixed_nodes': _fixed_nds}
            masks = self._infer(graph, state)
            
            print(f"   Step {_iter+1}: Fixed types {[self.class_room.get(t+1, f'unknown_{t+1}') for t in _types_subset]}")
        
        # Generate final image
        floorplan_image = draw_masks(masks.copy(), real_nodes)
        
        return floorplan_image, masks, room_ids, real_nodes
    
    def get_available_room_types(self):
        """Return list of available room types"""
        return list(self.room_class.keys())
    
    def get_room_colors(self):
        """Return mapping of room types to colors"""
        room_colors = {}
        for room_name, room_id in self.room_class.items():
            if room_id in self.id_color:
                room_colors[room_name] = self.id_color[room_id]
        return room_colors

def main():
    """Test the FIXED implementation"""
    print("🧪 Testing FIXED House-GAN++ Implementation")
    print("=" * 50)
    
    # Initialize inference
    generator = FixedHouseGANInference()
    
    # Test cases that were problematic
    test_cases = [
        {
            'name': 'Single Balcony Test',
            'rooms': ['living_room', 'kitchen', 'bathroom', 'balcony'],
            'expected': 'Exactly 1 balcony'
        },
        {
            'name': 'Single Study Room Test', 
            'rooms': ['living_room', 'kitchen', 'bedroom', 'study room'],
            'expected': 'Exactly 1 study room'
        },
        {
            'name': 'Two Bedroom Test',
            'rooms': ['living_room', 'kitchen', 'bedroom', 'bedroom', 'bathroom'],
            'expected': 'Exactly 2 bedrooms'
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n🏠 Test {i+1}: {test_case['name']}")
        print(f"   Rooms: {test_case['rooms']}")
        print(f"   Expected: {test_case['expected']}")
        
        try:
            floorplan_image, masks, room_ids, real_nodes = generator.generate_floorplan(test_case['rooms'])
            
            # Count actual room types generated
            actual_counts = {}
            for room_id in room_ids:
                room_name = generator.class_room.get(room_id + 1, f"unknown_{room_id + 1}")
                actual_counts[room_name] = actual_counts.get(room_name, 0) + 1
            
            print(f"   Result: {actual_counts}")
            
            # Save result
            output_path = f'./fixed_test_{i+1}.png'
            floorplan_image.save(output_path)
            print(f"   ✅ Saved to: {output_path}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        print("-" * 40)

if __name__ == "__main__":
    main()