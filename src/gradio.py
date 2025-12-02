import gradio as gr
import os
from src.utils.cmaps import spectra, spectra_warm
from pydub import AudioSegment  # for combining stems

cmap_mappings = {
    "Spectra (Cool)": spectra,
    "Spectra (Warm)": spectra_warm,
}

def ui():
    with gr.Blocks() as demo:
        gr.Markdown("### 🎶 Audio Spectrum Render")

        audio_input = gr.File(label="Upload Audio", file_types=[".wav", ".mp3"])

        # Fixed slots for 4 stems
        with gr.Row():
            render_output = gr.File(label="Output", file_types=[".mp4", ".mov"])
        

        # ---- Stem splitting ----
        def on_submit(file):
            from src.render import render_audio
            output_path = render_audio(path=file.name, opacity=0.7, translate_x=-7)
            print(output_path)
            return gr.Video(output_path)

        audio_input.change(
            fn=on_submit,
            inputs=audio_input,
            outputs=[
                render_output
            ],
        )
    return demo