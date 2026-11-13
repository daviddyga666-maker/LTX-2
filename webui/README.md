# LTX-2 Video Studio

A local Gradio web interface for the repository's **LTX-2.3 two-stage text/image-to-video pipeline**.

## Requirements

- Compatible NVIDIA CUDA GPU with substantial VRAM (the 22B model is very demanding)
- Python environment from the repository, installed with [uv](https://docs.astral.sh/uv/)
- Model weights downloaded separately: LTX-2.3 **dev** checkpoint, matching spatial upscaler, distilled LoRA, and Gemma 3 text encoder. See the repository root README.
- This interface is **local-only** and does not include hosted inference or free GPU compute.

## Setup

From the repository root:

```bash
uv sync --frozen
uv run --with "gradio>=5,<7" python webui/app.py
```

Open **http://127.0.0.1:7860** in your browser. On Windows, use PowerShell and run the same commands.

Enter the absolute paths to your downloaded model files in the UI, or configure environment variables `LTX_CHECKPOINT`, `LTX_GEMMA_ROOT`, `LTX_UPSCALER`, `LTX_DISTILLED_LORA`.

Describe the scene, optionally upload a starting image, choose output parameters, and click **Video generieren**. Outputs are saved in `outputs/webui/`.

## Notes

- Uses `python -m ltx_pipelines.ti2vid_two_stages`, not a simulated video generator.
- Output is a real MP4 with the pipeline's audio.
- Video dimensions must be divisible by 64; frame count must be `8*K+1`.
- Only one generation runs at a time; errors are shown in the UI.
- Do **not** expose port 7860 to the public internet. The local interface has no authentication and accepts model paths.
- GPU memory and model compatibility must be verified on the host computer.
