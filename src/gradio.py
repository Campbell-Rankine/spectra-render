import gradio as gr
import os
from pydub import AudioSegment  # for combining stems


def ui():
    with gr.Blocks() as demo:
        gr.Markdown("### 🎶 Audio Spectrum Render")

        audio_input = gr.File(label="Upload Audio", file_types=[".wav", ".mp3"])

        # Fixed slots for 4 stems
        with gr.Row():
            render_output = gr.Video(label="Output", type="filepath")



        # ---- Stem splitting ----
        def on_submit(file):
            from src.render import render_audio

            output_path = render_audio(path=file.name, opacity=0.7)

            labels, audios, files = splitter(file, output_path="./output")
            print(files, audios)
            del splitter

            # Map results back into fixed slots
            mapping = {
                "render": render_output,
            }

            render_vals = [output_path]

            return render_vals

        audio_input.change(
            fn=on_submit,
            inputs=audio_input,
            outputs=[
                render_output
            ],
        )
    return demo