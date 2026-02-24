"""
NOT FINISHED, WORK IN PROGRESS
TRYING TO FIGURE OUT HOW TO TRANSLATE REIDEMEISTER MOVES INTO SPATIAL TRASFORMATIONS of Beizer curve segments embedded in R3
"""

from typing import List, Dict, Tuple
import math

# --- basic data types ---
Segment = Dict   # {"id": int, "points": [(x,y), ...], "ori": +1/-1}
Crossing = Dict  # {"id":int, "over":seg_id, "under":seg_id, "pos_over":t, "pos_under":t, "sign":+1/-1}

class KnotDiagram:
    def __init__(self):
        self.segments: List[Segment] = []   # ordered segments
        self.crossings: List[Crossing] = [] # crossing table
        self.next_seg_id = 0
        self.next_x_id = 0

    def add_segment(self, points):
        seg = {"id": self.next_seg_id, "points": list(points), "ori": 1}
        self.next_seg_id += 1
        self.segments.append(seg)
        return seg["id"]

    def add_crossing(self, over_seg, under_seg, pos_over=0.5, pos_under=0.5, sign=1):
        x = {"id": self.next_x_id, "over":over_seg, "under":under_seg,
             "pos_over":pos_over, "pos_under":pos_under, "sign":sign}
        self.next_x_id += 1
        self.crossings.append(x)
        return x["id"]

    # --- simple helpers to find local neighborhood ---
    def find_adjacent_crossings(self, seg_id):
        return [c for c in self.crossings if c["over"]==seg_id or c["under"]==seg_id]

    # --- Reidemeister I: add/remove loop on a segment ---
    def apply_R1_add_loop(self, seg_id, pos=0.5, loop_size=0.02):
        """Add a tiny loop on segment seg_id at parameter pos (0..1)."""
        # create new small segment and a crossing (loop crossing)
        # For simplicity we don't reshape geometry here, just topology
        new_seg = self.add_segment([(0,0),(0,0)])  # placeholder coords
        self.add_crossing(over_seg=new_seg, under_seg=new_seg, pos_over=0.5, pos_under=0.5, sign=+1)
        return

    def apply_R1_remove_loop(self, crossing_id):
        """Remove a self-crossing that is an R1 loop (if found)."""
        # naive: remove crossing if over==under
        x = next((c for c in self.crossings if c["id"]==crossing_id), None)
        if x and x["over"]==x["under"]:
            self.crossings.remove(x)
        return

    # --- Reidemeister II: add/remove crossing pair between two segments ---
    def apply_R2_add_pair(self, segA, segB):
        """Create two opposite crossings between segA and segB."""
        self.add_crossing(over_seg=segA, under_seg=segB, pos_over=0.4, pos_under=0.4, sign=+1)
        self.add_crossing(over_seg=segB, under_seg=segA, pos_over=0.6, pos_under=0.6, sign=-1)
        return

    def apply_R2_remove_pair(self, crossing_id_1, crossing_id_2):
        """Remove two canceling crossings if they are an R2 pair."""
        c1 = next((c for c in self.crossings if c["id"]==crossing_id_1), None)
        c2 = next((c for c in self.crossings if c["id"]==crossing_id_2), None)
        if not c1 or not c2:
            return
        # naive check: same segment pair opposite order
        if c1["over"]==c2["under"] and c1["under"]==c2["over"]:
            self.crossings.remove(c1)
            self.crossings.remove(c2)
        return

    # --- Reidemeister III: slide crossing (topology-only) ---
    def apply_R3(self, crossing_triplet_ids: Tuple[int,int,int]):
        """Topological reconfiguration: implement as permutation of crossing partners."""
        # For clarity: this is highly schematic — R3 reassigns which strands cross which
        # A full robust R3 needs geometric checks. Here we provide a placeholder.
        return

    def __repr__(self):
        return f"KnotDiagram(segs={len(self.segments)}, x={len(self.crossings)})"
