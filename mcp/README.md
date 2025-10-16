# MCP Integration with TRELLIS

This guide explains how to integrate TRELLIS with Claude Code using the Model Context Protocol (MCP).

## What is MCP?

MCP (Model Context Protocol) allows you to extend Claude Code with custom tools and capabilities. In this case, we're creating an MCP server that can generate 2D images and convert them to 3D models.

## Setup Instructions

### 1. Install MCP

```bash
pip install mcp
```

### 2. Configure the MCP Server

Add the TRELLIS MCP server to your Claude Code configuration file at:
`~/.config/claude-code/settings.json`

```json
{
  "mcpServers": {
    "trellis": {
      "command": "python",
      "args": ["/home/vince/projects/IA/TRELLIS/mcp_server_example.py"]
    }
  }
}
```

### 3. Integrate with Image Generation

The example MCP server (`mcp_server_example.py`) has a placeholder for image generation. You need to integrate it with your preferred text-to-image model:

#### Option A: Stable Diffusion (Local)

```python
from diffusers import StableDiffusionPipeline

# In generate_2d_image function
sd_pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
sd_pipeline = sd_pipeline.to("cuda")
image = sd_pipeline(prompt).images[0]
return image
```

#### Option B: OpenAI DALL-E (API)

```python
import openai

# In generate_2d_image function
response = openai.Image.create(
    prompt=prompt,
    n=1,
    size=f"{width}x{height}"
)
image_url = response['data'][0]['url']
# Download and return image
```

#### Option C: ComfyUI (Local)

```python
import requests

# In generate_2d_image function
# Send request to ComfyUI API
response = requests.post('http://localhost:8188/prompt', json={
    'prompt': prompt,
    # ... ComfyUI workflow configuration
})
# Retrieve and return generated image
```

### 4. Restart Claude Code

After saving the MCP configuration, restart Claude Code to load the new MCP server.

## Usage from Claude Code

Once configured, you can use the MCP tools in Claude Code:

```
User: Generate a 3D model of a red sports car
Claude: [Uses mcp__trellis__text_to_3d tool to generate the model]
```

Available MCP tools:
- `mcp__trellis__generate_image` - Generate 2D image from text
- `mcp__trellis__image_to_3d` - Convert existing image to 3D
- `mcp__trellis__text_to_3d` - Complete pipeline: text → image → 3D

## Advanced: Custom Workflow

You can also create a custom workflow script:

```python
from mcp_server_example import generate_2d_image, image_to_glb

# Generate image
image = generate_2d_image("a cute robot")

# Convert to 3D
glb_data = image_to_glb(image, seed=42, texture_size=2048)

# Save or process the GLB file
with open("robot.glb", "wb") as f:
    f.write(base64.b64decode(glb_data))
```

## Troubleshooting

### MCP server not loading
- Check that the path in `settings.json` is correct
- Verify that `mcp` package is installed: `pip show mcp`
- Check Claude Code logs for errors

### Image generation not working
- The example uses a placeholder - you must integrate a real image generator
- Make sure your chosen image generation model is properly installed
- Check GPU memory availability

### 3D generation fails
- Ensure CUDA is available: `torch.cuda.is_available()`
- Check GPU memory (TRELLIS needs 16GB+ VRAM)
- Verify TRELLIS models are downloaded

## Alternative: REST API

If you prefer not to use MCP, you can create a simple REST API:

```python
from fastapi import FastAPI, UploadFile
import uvicorn

app = FastAPI()

@app.post("/generate-3d")
async def generate_3d(image: UploadFile):
    # Load image
    img = Image.open(image.file)

    # Generate 3D model
    # ... TRELLIS pipeline code ...

    return {"glb_url": "path/to/generated.glb"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Then call it from anywhere:
```bash
curl -X POST -F "image=@photo.png" http://localhost:8000/generate-3d
```