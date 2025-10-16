#!/usr/bin/env python3
"""
MCP Server example for generating 2D images and converting to 3D with TRELLIS.

This is an example MCP (Model Context Protocol) server that provides tools to:
1. Generate 2D images (placeholder - integrate with your favorite image generator)
2. Convert those images to 3D models using TRELLIS

To use this with Claude Code:
1. Install mcp: pip install mcp
2. Add to your Claude Code MCP settings in ~/.config/claude-code/settings.json:

{
  "mcpServers": {
    "trellis": {
      "command": "python",
      "args": ["/home/vince/projects/IA/TRELLIS/mcp_server_example.py"]
    }
  }
}

Then you can use it from Claude Code with tools like:
- mcp__trellis__generate_image
- mcp__trellis__image_to_3d
- mcp__trellis__text_to_3d
- mcp__trellis__image_to_3d_only_mesh
"""

import os
import sys
import asyncio
import base64
from io import BytesIO
from typing import Optional
from PIL import Image
import trimesh

# MCP imports
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
except ImportError:
    print("Error: mcp package not installed. Install with: pip install mcp", file=sys.stderr)
    sys.exit(1)

# TRELLIS imports
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils

# Environment setup
os.environ['SPCONV_ALGO'] = 'native'

# Global pipeline instance (loaded once)
pipeline: Optional[TrellisImageTo3DPipeline] = None

def load_pipeline():
    """Load TRELLIS pipeline (lazy loading)"""
    global pipeline
    if pipeline is None:
        pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
        pipeline.cuda()
    return pipeline

def generate_2d_image(prompt: str, width: int = 512, height: int = 512) -> Image.Image:
    """
    Generate a 2D image from a text prompt.

    TODO: Integrate with your preferred image generation model:
    - Stable Diffusion
    - DALL-E API
    - Midjourney API
    - Any other text-to-image model

    For now, this returns a placeholder image.
    """
    # PLACEHOLDER: Replace with actual image generation
    print(f"Generating image with prompt: {prompt}", file=sys.stderr)

    # Example: Create a simple colored square as placeholder
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new('RGB', (width, height), color='lightblue')
    draw = ImageDraw.Draw(img)

    # Add text
    text = f"TODO: {prompt[:50]}"
    draw.text((10, 10), text, fill='black')

    return img

def image_to_mesh(image: Image.Image, output_path: str, seed: int = 1, simplify: float = 0.95, format: str = "obj") -> str:
    """
    Convert image to 3D mesh (without texture) and export in specified format.

    Supported formats: obj, ply, stl, glb (with basic gray material)
    """
    print("Converting image to 3D mesh...", file=sys.stderr)

    # Load pipeline
    pipe = load_pipeline()

    # Generate 3D model (mesh only, skip Gaussian)
    outputs = pipe.run(
        image,
        seed=seed,
        formats=["mesh"],
    )

    # Get the mesh
    mesh_result = outputs['mesh'][0]

    # Postprocess mesh (simplify and clean)
    processed_mesh = postprocessing_utils.postprocess_mesh(
        mesh_result,
        simplify=simplify,
    )

    # Create trimesh object without texture
    mesh_trimesh = trimesh.Trimesh(
        vertices=processed_mesh.vertices.cpu().numpy(),
        faces=processed_mesh.faces.cpu().numpy(),
    )

    # Export based on format
    if format == 'obj':
        mesh_trimesh.export(output_path, file_type='obj')
    elif format == 'ply':
        mesh_trimesh.export(output_path, file_type='ply')
    elif format == 'stl':
        mesh_trimesh.export(output_path, file_type='stl')
    elif format == 'glb':
        # Export GLB with basic gray material (no texture baking)
        mesh_trimesh.visual = trimesh.visual.ColorVisuals(
            mesh=mesh_trimesh,
            vertex_colors=[128, 128, 128, 255]  # Gray color
        )
        mesh_trimesh.export(output_path, file_type='glb')

    return f"Mesh generated and saved to: {output_path}"

# Create MCP server
app = Server("trellis")

@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools"""
    return [
        Tool(
            name="generate_image",
            description="Generate a 2D image from a text prompt (placeholder - integrate with your image generator)",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Text description of the image to generate"
                    },
                    "width": {
                        "type": "integer",
                        "description": "Image width in pixels",
                        "default": 512
                    },
                    "height": {
                        "type": "integer",
                        "description": "Image height in pixels",
                        "default": 512
                    }
                },
                "required": ["prompt"]
            }
        ),
        Tool(
            name="image_to_3d",
            description="Convert a 2D image to a 3D GLB model using TRELLIS",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Path to the input image file"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path where the GLB file should be saved"
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Random seed for generation",
                        "default": 1
                    },
                    "texture_size": {
                        "type": "integer",
                        "description": "Texture resolution (512, 1024, or 2048)",
                        "default": 1024
                    }
                },
                "required": ["image_path", "output_path"]
            }
        ),
        Tool(
            name="text_to_3d",
            description="Generate a 2D image from text, then convert to 3D model",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Text description of the 3D object to create"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path where the GLB file should be saved"
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Random seed for generation",
                        "default": 1
                    }
                },
                "required": ["prompt", "output_path"]
            }
        ),
        Tool(
            name="image_to_3d_only_mesh",
            description="Convert a 2D image to a 3D mesh (no texture) using TRELLIS. Supports OBJ, PLY, STL, and GLB formats",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Path to the input image file"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path where the mesh file should be saved"
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Random seed for generation",
                        "default": 1
                    },
                    "simplify": {
                        "type": "number",
                        "description": "Mesh simplification ratio (0.0-1.0, higher = more simplified)",
                        "default": 0.95
                    },
                    "format": {
                        "type": "string",
                        "description": "Output format (obj, ply, stl, or glb)",
                        "enum": ["obj", "ply", "stl", "glb"],
                        "default": "obj"
                    }
                },
                "required": ["image_path", "output_path"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls"""

    if name == "generate_image":
        prompt = arguments["prompt"]
        width = arguments.get("width", 512)
        height = arguments.get("height", 512)

        # Generate image
        image = generate_2d_image(prompt, width, height)

        # Save to temp file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            image.save(tmp.name)
            temp_path = tmp.name

        return [
            TextContent(
                type="text",
                text=f"Generated image saved to: {temp_path}\n"
                     f"(Note: This is a placeholder. Integrate with a real image generator.)"
            )
        ]

    elif name == "image_to_3d":
        image_path = arguments["image_path"]
        output_path = arguments["output_path"]
        seed = arguments.get("seed", 1)
        texture_size = arguments.get("texture_size", 1024)

        # Load image
        image = Image.open(image_path)

        # Convert to 3D
        pipe = load_pipeline()
        outputs = pipe.run(
            image,
            seed=seed,
            formats=["gaussian", "mesh"],
        )

        # Extract GLB
        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            simplify=0.95,
            texture_size=texture_size,
        )

        # Export
        glb.export(output_path)

        return [
            TextContent(
                type="text",
                text=f"3D model generated and saved to: {output_path}"
            )
        ]

    elif name == "text_to_3d":
        prompt = arguments["prompt"]
        output_path = arguments["output_path"]
        seed = arguments.get("seed", 1)

        # Generate 2D image
        image = generate_2d_image(prompt)

        # Convert to 3D
        pipe = load_pipeline()
        outputs = pipe.run(
            image,
            seed=seed,
            formats=["gaussian", "mesh"],
        )

        # Extract GLB
        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            simplify=0.95,
            texture_size=1024,
        )

        # Export
        glb.export(output_path)

        return [
            TextContent(
                type="text",
                text=f"3D model generated from prompt '{prompt}' and saved to: {output_path}\n"
                     f"(Note: Using placeholder image generator. Integrate with a real model for better results.)"
            )
        ]

    elif name == "image_to_3d_only_mesh":
        image_path = arguments["image_path"]
        output_path = arguments["output_path"]
        seed = arguments.get("seed", 1)
        simplify = arguments.get("simplify", 0.95)
        file_format = arguments.get("format", "obj")

        # Load image
        image = Image.open(image_path)

        # Convert to mesh
        result_message = image_to_mesh(image, output_path, seed, simplify, file_format)

        return [
            TextContent(
                type="text",
                text=result_message
            )
        ]

    else:
        raise ValueError(f"Unknown tool: {name}")

async def main():
    """Run the MCP server"""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
