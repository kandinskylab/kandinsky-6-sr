"""Explicit model downloads from the shipped workflow manifest; no startup network calls."""

from __future__ import annotations

import asyncio
import errno
import json
import logging
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from aiohttp import web
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

_MODEL_DIRS = {"diffusion_models", "text_encoders", "vae", "audio_vae"}
_MIN_HF_URL_PARTS = 5  # owner/repo/resolve/main/file


def model_plan(workflows: Path) -> list[dict[str, str]]:
    entries = {}
    for workflow in sorted(workflows.glob("*.json")):
        graph = json.loads(workflow.read_text(encoding="utf-8"))
        for node in graph["nodes"]:
            for field in ("models", "kandinsky6_required_files"):
                for item in node.get("properties", {}).get(field, []):
                    name = PurePosixPath(item["name"])
                    url = urlsplit(item["url"])
                    parts = url.path.strip("/").split("/")
                    if (
                        item["directory"] not in _MODEL_DIRS
                        or name.is_absolute()
                        or ".." in name.parts
                        or "\\" in item["name"]
                        or url.scheme != "https"
                        or url.netloc != "huggingface.co"
                        or url.query
                        or url.fragment
                        or len(parts) < _MIN_HF_URL_PARTS
                        or parts[2:4] != ["resolve", "main"]
                        or any(part in ("", ".", "..") for part in parts)
                    ):
                        raise ValueError("Invalid packaged model manifest")
                    key = item["directory"] + "/" + item["name"]
                    entry = {**item, "repo_id": "/".join(parts[:2]), "filename": "/".join(parts[4:])}
                    if key in entries and entries[key] != entry:
                        raise ValueError("Conflicting packaged model destinations")
                    entries[key] = entry
    if not entries:
        raise ValueError("Empty packaged model manifest")
    return list(entries.values())


def existing_model(entry, models_root, extra_roots):
    roots = [models_root / entry["directory"], *extra_roots.get(entry["directory"], [])]
    for root in roots:
        candidate = Path(root) / entry["name"]
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def download_model(entry, models_root, extra_roots):
    if existing_model(entry, models_root, extra_roots) is not None:
        return "reused"
    category = models_root / entry["directory"]
    destination = category / entry["name"]
    if not destination.parent.resolve().is_relative_to(category.resolve()):
        raise ValueError("Incomplete model bundle points outside its configured folder")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Download on the destination filesystem: resumable partials, then an atomic
    # move into the exact Comfy filename, without a second full copy of the weights.
    downloaded = Path(
        hf_hub_download(
            repo_id=entry["repo_id"],
            filename=entry["filename"],
            revision="main",
            local_dir=destination.parent / ".kandinsky6-downloads" / entry["repo_id"].replace("/", "--"),
        )
    )
    if existing_model(entry, models_root, extra_roots) is not None:
        return "reused"
    downloaded.replace(destination)
    return "downloaded"


def download_error(error):
    if isinstance(error, HfHubHTTPError) and error.response is not None and error.response.status_code in (401, 403):
        return "Hugging Face access required. Run hf auth login on this server, then retry."
    if isinstance(error, OSError) and error.errno == errno.ENOSPC:
        return "Not enough free disk space on the ComfyUI server."
    if isinstance(error, PermissionError):
        return "Cannot write to the ComfyUI models folder. Check folder permissions."
    return "Download failed. Check the ComfyUI console and retry; partial downloads can resume."


def register_routes(routes, package_id, workflows, models_root, extra_roots):
    plan = model_plan(workflows)
    lock = asyncio.Lock()

    @routes.get(f"/{package_id}/models")
    async def get_models(request):
        return web.json_response(
            {
                "models": [
                    {
                        "path": item["directory"] + "/" + item["name"],
                        "ready": existing_model(item, models_root, extra_roots) is not None,
                    }
                    for item in plan
                ],
            }
        )

    @routes.post(f"/{package_id}/download-models")
    async def download_models(request):
        if request.content_type != "application/json":
            raise web.HTTPBadRequest(text="Explicit JSON confirmation is required.")
        try:
            payload = await request.json()
        except json.JSONDecodeError as error:
            raise web.HTTPBadRequest(text="Invalid JSON confirmation.") from error
        if payload != {"confirm": True}:
            raise web.HTTPBadRequest(text="Explicit download confirmation is required.")
        if lock.locked():
            raise web.HTTPConflict(text="A download is already running for this package.")
        async with lock:
            response = web.StreamResponse(
                headers={
                    "Content-Type": "application/x-ndjson",
                    "Cache-Control": "no-store",
                    "X-Accel-Buffering": "no",
                }
            )
            await response.prepare(request)
            task = None

            async def send(event):
                await response.write((json.dumps(event) + "\n").encode())

            try:
                for index, entry in enumerate(plan, 1):
                    path = entry["directory"] + "/" + entry["name"]
                    event = {"path": path, "index": index, "total": len(plan)}
                    await send({**event, "status": "checking"})
                    task = asyncio.create_task(asyncio.to_thread(download_model, entry, models_root, extra_roots))
                    while not task.done():
                        await asyncio.wait({task}, timeout=5)
                        if not task.done():
                            await send({**event, "status": "downloading"})
                    await send({**event, "status": await task})
                await send({"status": "complete", "total": len(plan)})
            except (ConnectionResetError, BrokenPipeError):
                # Finish only the already-authorized current file, not the rest.
                pass
            except Exception as error:
                logging.exception("Kandinsky model download failed")
                await send({"status": "error", "message": download_error(error)})
            finally:
                if task is not None and not task.done():
                    await asyncio.shield(task)
            return response
