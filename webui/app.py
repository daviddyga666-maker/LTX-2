"""Local Gradio interface for LTX-2.3 inference.

Run from repository root: uv run --with gradio python webui/app.py
"""
import os
import subprocess
import sys
import threading
import uuid
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs" / "webui"
UPLOADS = OUTPUTS / "uploads"
OUTPUTS.mkdir(parents=True, exist_ok=True)
UPLOADS.mkdir(parents=True, exist_ok=True)
LOCK = threading.Lock()


def generate(prompt, image, checkpoint, gemma, upscaler, lora, width, height, frames, fps, steps, seed, strength, quantization, progress=gr.Progress()):
    if not prompt or not prompt.strip():
        raise gr.Error("Bitte einen Videoprompt eingeben.")
    paths = {"Checkpoint": checkpoint, "Gemma-Verzeichnis": gemma, "Spatial Upscaler": upscaler, "Distilled LoRA": lora}
    for label, value in paths.items():
        if not value or not Path(value).expanduser().exists():
            raise gr.Error(f"{label} fehlt oder wurde nicht gefunden: {value}")
    width, height, frames, steps = int(width), int(height), int(frames), int(steps)
    if width % 64 or height % 64 or width < 256 or height < 256:
        raise gr.Error("Breite und Höhe müssen mindestens 256 und durch 64 teilbar sein.")
    if frames < 9 or (frames - 1) % 8:
        raise gr.Error("Frame-Anzahl muss 8 × K + 1 sein (z. B. 49, 97, 121).")
    if not 1 <= steps <= 100 or not 1 <= float(fps) <= 60:
        raise gr.Error("Ungültige Schrittzahl oder Bildrate.")
    if not 0 <= float(strength) <= 1:
        raise gr.Error("Bildstärke muss zwischen 0 und 1 liegen.")
    if not LOCK.acquire(blocking=False):
        raise gr.Error("Eine Generierung läuft bereits. Bitte warten.")
    try:
        output = OUTPUTS / f"video-{uuid.uuid4().hex[:12]}.mp4"
        cmd = [
            sys.executable, "-m", "ltx_pipelines.ti2vid_two_stages",
            "--checkpoint-path", str(Path(checkpoint).expanduser().resolve()),
            "--gemma-root", str(Path(gemma).expanduser().resolve()),
            "--spatial-upsampler-path", str(Path(upscaler).expanduser().resolve()),
            "--distilled-lora", str(Path(lora).expanduser().resolve()), "0.8",
            "--prompt", prompt.strip(), "--output-path", str(output),
            "--width", str(width), "--height", str(height),
            "--num-frames", str(frames), "--frame-rate", str(fps),
            "--num-inference-steps", str(steps), "--seed", str(int(seed)),
        ]
        if image:
            source = Path(image).resolve()
            if source.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                raise gr.Error("Bitte PNG, JPG oder WEBP verwenden.")
            # Gradio's upload is already stored on disk. The CLI resolves the path.
            cmd += ["--image", str(source), "0", str(float(strength))]
        if quantization != "Keine":
            cmd += ["--quantization", quantization]
        progress(0, desc="LTX-2 wird ausgeführt. Die erste Generierung kann länger dauern.")
        env = os.environ.copy()
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=7200)
        if proc.returncode != 0:
            details = (proc.stderr or proc.stdout)[-4500:]
            raise gr.Error(f"Generierung fehlgeschlagen:\n{details}")
        if not output.is_file():
            raise gr.Error("Die Pipeline hat keine MP4-Datei erzeugt.")
        progress(1, desc="Fertig")
        return str(output), str(output), f"Video erstellt: {output.name}"
    except subprocess.TimeoutExpired:
        raise gr.Error("Zeitlimit von zwei Stunden überschritten.")
    finally:
        LOCK.release()


with gr.Blocks(title="LTX-2 Video Studio", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# LTX-2 Video Studio\n**Text-to-Video und Image-to-Video mit deinem lokalen LTX-2.3-Modell.**")
    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(label="Videobeschreibung", lines=5, placeholder="A cinematic tracking shot of a futuristic city at night...")
            image = gr.Image(label="Optionales Startbild", type="filepath", sources=["upload"])
            with gr.Accordion("Modellpfade (lokal)", open=True):
                checkpoint = gr.Textbox(label="LTX-2.3 Dev Checkpoint", value=os.getenv("LTX_CHECKPOINT", ""))
                gemma = gr.Textbox(label="Gemma 3 Modellordner", value=os.getenv("LTX_GEMMA_ROOT", ""))
                upscaler = gr.Textbox(label="Spatial Upscaler", value=os.getenv("LTX_UPSCALER", ""))
                lora = gr.Textbox(label="Distilled LoRA", value=os.getenv("LTX_DISTILLED_LORA", ""))
            with gr.Accordion("Videoeinstellungen", open=False):
                with gr.Row():
                    width = gr.Dropdown([512, 768, 1024, 1280], value=768, label="Breite")
                    height = gr.Dropdown([512, 768, 1024, 1280], value=512, label="Höhe")
                frames = gr.Dropdown([49, 97, 121, 161, 193], value=97, label="Frames (8 × K + 1)")
                fps = gr.Slider(8, 30, value=24, step=1, label="FPS")
                steps = gr.Slider(10, 60, value=30, step=1, label="Inference-Schritte")
                seed = gr.Number(value=42, precision=0, label="Seed")
                strength = gr.Slider(0, 1, value=0.8, step=0.05, label="Startbild-Stärke")
                quantization = gr.Dropdown(["Keine", "fp8-cast"], value="Keine", label="Quantisierung")
            submit = gr.Button("Video generieren", variant="primary")
        with gr.Column(scale=1):
            video = gr.Video(label="Ergebnis", interactive=False)
            download = gr.File(label="MP4 herunterladen")
            status = gr.Textbox(label="Status", interactive=False)
    submit.click(
        fn=generate,
        inputs=[prompt, image, checkpoint, gemma, upscaler, lora, width, height, frames, fps, steps, seed, strength, quantization],
        outputs=[video, download, status],
        concurrency_limit=1,
    )

if __name__ == "__main__":
    demo.queue(max_size=4).launch(server_name="127.0.0.1", server_port=7860, show_error=True, allowed_paths=[str(OUTPUTS)])
