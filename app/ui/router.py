from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

ui = APIRouter(prefix="/ui", tags=["ui"])
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


@ui.get("/console", response_class=HTMLResponse)
async def console(request: Request):
    """Web console for testing RAG Patient API"""
    return templates.TemplateResponse("console.html", {"request": request})
