from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import requests

from app.routes import search_route

app = FastAPI(title="Image Search API")

# Include API router
app.include_router(search_route.router, prefix="/api")

# Mount static files & templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# UI: Giao diện web
@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# UI: Upload ảnh từ form
@app.post("/search_local", response_class=HTMLResponse)
async def search_local(request: Request, file: UploadFile = File(...)):
    files = {"file": (file.filename, await file.read(), file.content_type)}
    response = requests.post("http://127.0.0.1:8000/api/search", files=files)

    data = response.json()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "results": data.get("results", []),
        "message": data.get("message", "")
    })
