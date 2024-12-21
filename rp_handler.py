import runpod
import torch
from PIL import Image
import os
from datetime import datetime
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
import base64
import io
import numpy as np

# Initialize pipeline
def init_pipeline():
    pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
    pipeline.cuda()
    return pipeline

# Global pipeline instance
trellis_pipe = init_pipeline()

def handler(event):
    try:
        input_data = event["input"]
        image_base64 = input_data.get("image_base64")
        mesh_simplify = input_data.get("mesh_simplify", 0.95)
        texture_size = input_data.get("texture_size", 1024)
        
        # Add missing parameters from headless version
        seed = input_data.get("seed", 0)
        randomize_seed = input_data.get("randomize_seed", True)
        ss_guidance_strength = input_data.get("ss_guidance_strength", 7.5)
        ss_sampling_steps = input_data.get("ss_sampling_steps", 12)
        slat_guidance_strength = input_data.get("slat_guidance_strength", 3.0)
        slat_sampling_steps = input_data.get("slat_sampling_steps", 12)

        if not image_base64:
            return {"error": "Image base64 data not provided"}
        
        try:
            # Decode base64 to image
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
        except Exception as e:
            return {"error": f"Failed to decode base64 image: {str(e)}"}
        
        # Update seed if randomization is requested
        if randomize_seed:
            seed = np.random.randint(0, np.iinfo(np.int32).max)
        
        # Generate 3D model
        outputs = trellis_pipe.run(
            image,
            seed=seed,
            formats=["gaussian", "mesh"],
            sparse_structure_sampler_params={
                "steps": ss_sampling_steps,
                "cfg_strength": ss_guidance_strength,
            },
            slat_sampler_params={
                "steps": slat_sampling_steps,
                "cfg_strength": slat_guidance_strength,
            }
        )
        
        # Generate GLB file
        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            simplify=mesh_simplify,
            texture_size=texture_size
        )
        
        # Save GLB to bytes buffer
        buffer = io.BytesIO()
        glb.export(buffer)
        
        # Convert to base64
        glb_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return {
            "glb_base64": glb_base64
        }
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler}) 