import torch
import numpy as np
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils
import io
import base64

class ModelGenerator:
    def __init__(self):
        self.pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
        self.pipeline.cuda()
        self.MAX_SEED = np.iinfo(np.int32).max

    def generate(self, image, seed=0, randomize_seed=True,
                ss_guidance_strength=7.5, ss_sampling_steps=12,
                slat_guidance_strength=3.0, slat_sampling_steps=12,
                mesh_simplify=0.95, texture_size=1024):

        print(f"Generating 3D model with seed: {seed}")
        
        # Update seed if randomization is requested
        if randomize_seed:
            seed = np.random.randint(0, self.MAX_SEED)
        
        # Generate 3D model
        outputs = self.pipeline.run(
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
        
        # Save GLB to bytes buffer and convert to base64
        buffer = io.BytesIO()
        glb.export(buffer, format="glb")
        glb_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return {
            "glb_base64": glb_base64,
            "seed": seed
        } 