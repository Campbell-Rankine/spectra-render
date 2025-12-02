import sys
import os
import numpy as np
from timeit import default_timer as timer
import argparse

from manim import config, tempconfig
from src.utils.logging import init_logger
from src.utils.audio import AudioIO
from src.utils.fft_hist import FFT_Histogram

# only to be used on docker

def render_audio(path: str, opacity: float, translate_x=0, translate_y=-0.5, translate_z=0):
    start = timer()
    logger = init_logger()

    if not os.path.exists(path):
        raise ValueError(f"File does not exist at path: {path}")

    output_file = f"{path.split('/')[-1].split('.')[0]}-render"
    output_path = f"./output/video/{output_file}.mp4"
    logger.info(f"Rendering file at: {path}, Output location: {output_file}")

    with tempconfig(
        {
            "quality": "medium_quality",
            "output_file": output_file,
            "format": "mp4",
            "media_dir": "./output",  # parent folder for videos/images
            "video_dir": "./output/video",  # subfolder for rendered videos
            "images_dir": "./output/frames",
        }
    ):
        # construct scene object
        scene = FFT_Histogram(
            path=path,
            num_bins=512,
            log_scale=True,
            log_base=2,
            translate_x=translate_x,
            translate_y=translate_y,
            translate_z=translate_z,
            bar_width=0.05,
            height_clipping=2.75,
            opacity=opacity,
        )
        scene.register("logger", logger)

        # render
        scene.render()

    # attach audio to video
    end = timer()
    logger.info(f"Rendering Scene Took {round(end-start, 2)} seconds")
    return output_path