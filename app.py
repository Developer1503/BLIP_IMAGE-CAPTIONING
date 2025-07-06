from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import Blip2ForConditionalGeneration, AutoProcessor
from PIL import Image
import torch
import io
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load BLIP-2 model and processor
try:
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    logger.info("BLIP-2 model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load BLIP-2 model: {e}")
    raise

# Initialize FastAPI
app = FastAPI(
    title="BLIP-2 Image Captioning API",
    description="Upload an image and receive an AI-generated caption.",
    version="1.0.0"
)

@app.post("/caption")
async def generate_caption(file: UploadFile = File(...)):
    """
    Upload an image and receive a caption.
    """
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Prepare input
        vision_inputs = processor(images=image, return_tensors="pt").to(device)
        prompt = "Describe the image."
        text_inputs = processor(text=prompt, return_tensors="pt").to(device)
        inputs = {**vision_inputs, **text_inputs}

        # Generate caption
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=50)
            caption = processor.decode(generated_ids[0], skip_special_tokens=True)

        return {"caption": caption}

    except Exception as e:
        logger.error(f"Captioning error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
