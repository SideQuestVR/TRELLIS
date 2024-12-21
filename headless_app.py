import base64
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from .model_generator import ModelGenerator
import io
from pydantic import BaseModel

# Add new request model
class ImageRequest(BaseModel):
    image_base64: str
    seed: int = 0
    randomize_seed: bool = True
    ss_guidance_strength: float = 7.5
    ss_sampling_steps: int = 12
    slat_guidance_strength: float = 3.0
    slat_sampling_steps: int = 12
    mesh_simplify: float = 0.95
    texture_size: int = 1024

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize generator globally
generator = ModelGenerator()

@app.post("/process-image")
async def process_image(request: ImageRequest):
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_data))
        
        # Generate 3D model
        return generator.generate(
            image=image,
            seed=request.seed,
            randomize_seed=request.randomize_seed,
            ss_guidance_strength=request.ss_guidance_strength,
            ss_sampling_steps=request.ss_sampling_steps,
            slat_guidance_strength=request.slat_guidance_strength,
            slat_sampling_steps=request.slat_sampling_steps,
            mesh_simplify=request.mesh_simplify,
            texture_size=request.texture_size
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
