import cv2
import mujoco as mj
import mujoco_viewer as mjv
import numpy as np
import pathlib
from dataclasses import dataclass

@dataclass
class RenderOptions:
    width: int = 800
    height: int = 600
    fps: int = 30

class MujocoVideoRecorder:
    def __init__(
        self,
        model: mj.MjModel,
        data: mj.MjData,
        options: RenderOptions,
    ):
        self.model: self.MjModel = model
        self.data: mj.MjData = data
        self.width: int = options.width
        self.height: int = options.height
        self.fps: int = options.fps

        self.viewer = mjv.MujocoViewer(self.model, self.data, 'offscreen')
        self.frames = []

    def reset(self):
        self.frames = []

    def record_frame(self):
        img = self.viewer.read_pixels()
        self.frames.append(img)

    def write_video(self, video_path: str | pathlib.Path):
        print("0")
        video_path = str(video_path)
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (self.width, self.height))

        print("1")
        for frame in self.frames:
            frame = cv2.resize(frame,(self.width,self.height))
            writer.write(frame)

        print("2")
        writer.release()