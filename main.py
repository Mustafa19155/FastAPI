# from fastapi import FastAPI

# app = FastAPI()

# @app.get("/")
# async def root():
#     return {"greeting": "Hello, World!", "message": "Welcome to FastAPI!"}

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from io import BytesIO
import base64

# Import the Cloudinary SDK
import cloudinary
import cloudinary.uploader

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

device = "cuda"
model_id = "dreamlike-art/dreamlike-photoreal-2.0"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_auth_token="hf_FIoLXhGCaejfrtW")
pipe.to(device)

# Initialize Cloudinary (Replace 'your_cloud_name', 'your_api_key', and 'your_api_secret' with your Cloudinary credentials)
cloudinary.config(
    cloud_name="dgu",
    api_key="569428144",
    api_secret="N-t0PNEwA"
)

@app.get("/")
async def generate(prompt: str, height: int, width: int):
    with autocast(device):
        image = pipe(prompt, guidance_scale=8.5, height=height, width=width).images[0]

    image.save("testimage.png")
    buffer = BytesIO()
    image.save(buffer, format="PNG")

    # Upload the image to Cloudinary
    upload_result = cloudinary.uploader.upload(buffer.getvalue(), folder="Graphit")

    # Get the URL of the uploaded image from the Cloudinary response
    image_url = upload_result["secure_url"]

    return {"image_url": image_url}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# from fastapi import FastAPI, Response
# from fastapi.middleware.cors import CORSMiddleware
# import torch
# from torch import autocast
# from diffusers import StableDiffusionPipeline
# from io import BytesIO
# import base64


# # Import the Cloudinary SDK
# import cloudinary
# import cloudinary.uploader


# import nest_asyncio
# from pyngrok import ngrok
# import uvicorn

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_credentials=True,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"]
# )


# device = "cuda"
# model_id = "dreamlike-art/dreamlike-photoreal-2.0"
# pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16,use_auth_token="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")
# pipe.to(device)

# # Initialize Cloudinary (Replace 'your_cloud_name', 'your_api_key', and 'your_api_secret' with your Cloudinary credentials)
# cloudinary.config(
#     cloud_name="dgusq1sv0",
#     api_key="569428144272248",
#     api_secret="N-t0PNEw-adXc4uNH3O2oiuJEoA"
# )

# @app.get("/")
# async def generate(prompt: str, height: int, width: int):
#     with autocast(device):
#         image = pipe(prompt, guidance_scale=8.5, height= height, width= width).images[0]
#         # print(prompt)
#         # print(height)
#         # print(width)

#     image.save("testimage.png")
#     buffer = BytesIO()
#     image.save(buffer, format="PNG")

#     # Upload the image to Cloudinary
#     upload_result = cloudinary.uploader.upload(buffer.getvalue(), folder="Graphit")

#       # Get the URL of the uploaded image from the Cloudinary response
#     image_url = upload_result["secure_url"]


#     return {"image_url": image_url}


# ngrok_tunnel = ngrok.connect(8000)
# print('Public URL:', ngrok_tunnel.public_url)
# nest_asyncio.apply()
# uvicorn.run(app,port=8000)
