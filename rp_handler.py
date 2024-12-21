import runpod
from PIL import Image
import base64
import io
from model_generator import ModelGenerator

# Initialize generator globally
generator = ModelGenerator()

def handler(event):
    try:
        input_data = event["input"]
        
        # Extract parameters from input
        image_base64 = input_data.get("image_base64")
        if not image_base64:
            return {"error": "Image base64 data not provided"}
        
        try:
            # Decode base64 to image
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
        except Exception as e:
            return {"error": f"Failed to decode base64 image: {str(e)}"}
        
        # Generate 3D model using ModelGenerator
        return generator.generate(
            image=image,
            seed=input_data.get("seed", 0),
            randomize_seed=input_data.get("randomize_seed", True),
            ss_guidance_strength=input_data.get("ss_guidance_strength", 7.5),
            ss_sampling_steps=input_data.get("ss_sampling_steps", 12),
            slat_guidance_strength=input_data.get("slat_guidance_strength", 3.0),
            slat_sampling_steps=input_data.get("slat_sampling_steps", 12),
            mesh_simplify=input_data.get("mesh_simplify", 0.95),
            texture_size=input_data.get("texture_size", 1024)
        )
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler}) 