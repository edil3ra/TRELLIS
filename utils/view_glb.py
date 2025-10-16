#!/usr/bin/env python3
"""Simple GLB viewer using trimesh and pyglet"""
import sys
import trimesh

if len(sys.argv) < 2:
    print("Usage: python view_glb.py <path_to_glb_file>")
    sys.exit(1)

glb_file = sys.argv[1]
print(f"Loading {glb_file}...")

# Load the GLB file
mesh = trimesh.load(glb_file)

# Show the mesh in an interactive viewer
print("Opening viewer... Close the window when done.")
mesh.show()