import pyrealsense2 as rs
import numpy as np
import cv2
import time
import torch
from ultralytics import YOLO
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from PIL import Image
import re

class ASFMNavigator:
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(config)

        self.yolo = YOLO('yolov8n.pt').to('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
        self.vlm_model = InstructBlipForConditionalGeneration.from_pretrained(
            "Salesforce/instructblip-vicuna-7b"
        ).to('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.personal_space = 1.2
        self.min_speed = 0.2
        self.max_speed = 1.5

    def deproject(self, depth_frame, pixel):
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        depth = depth_frame.get_distance(int(pixel[0]), int(pixel[1]))
        if depth == 0:
            return None
        point = rs.rs2_deproject_pixel_to_point(depth_intrin, [pixel[0], pixel[1]], depth)
        return np.array(point)

    def detect_humans(self, color_frame, depth_frame):
        results = self.yolo(color_frame)[0]
        humans = []
        for box in results.boxes:
            if int(box.cls[0]) == 0:  # person class
                bbox = box.xyxy[0].cpu().numpy()
                cx = int((bbox[0] + bbox[2]) / 2)
                cy = int((bbox[1] + bbox[3]) / 2)
                pos_3d = self.deproject(depth_frame, (cx, cy))
                if pos_3d is not None and pos_3d[2] > 0:
                    humans.append({'position': pos_3d, 'bbox': bbox})
        return humans

    def analyze_environment(self, depth_image, humans):
        h, w = depth_image.shape
        left = np.nanmean(depth_image[:, :w//3])
        center = np.nanmean(depth_image[:, w//3:2*w//3])
        right = np.nanmean(depth_image[:, 2*w//3:])

        free_space = {
            'left': left > 1.0,
            'center': center > 1.0,
            'right': right > 1.0
        }

        situation_text = f"Free space - Left: {'Yes' if free_space['left'] else 'No'}, Center: {'Yes' if free_space['center'] else 'No'}, Right: {'Yes' if free_space['right'] else 'No'}. "
        if not humans:
            situation_text += "No humans detected."

        return situation_text, free_space

    def generate_vlm_command(self, color_frame, situation_text):
        prompt = (
            f"Situation: {situation_text} "
            f"Suggest the best movement action and safe speed (0.2-1.5m/s). "
            f"Actions: Move_Left, Move_Right, Move_Straight, Stop"
        )
        inputs = self.processor(
            images=Image.fromarray(cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)),
            text=prompt,
            return_tensors="pt"
        ).to(self.device)
        outputs = self.vlm_model.generate(**inputs, max_new_tokens=50)
        decoded = self.processor.decode(outputs[0], skip_special_tokens=True)

        match = re.search(r"(Move_Left|Move_Right|Move_Straight|Stop).*?(\d+(\.\d+)?)m/s", decoded)
        if match:
            action = match.group(1)
            speed = float(match.group(2))
            return action, speed
        return "Move_Straight", 1.0

    def visualize(self, frame, humans, command, situation_text):
        for human in humans:
            x1, y1, x2, y2 = map(int, human['bbox'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{situation_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, f"Command: {command}", (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return frame

    def run(self):
        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32)
                depth_image[depth_image == 0] = np.nan

                humans = self.detect_humans(color_image, depth_frame)
                situation_text, free_space = self.analyze_environment(depth_image, humans)
                print("[Situation]", situation_text)

                if not free_space['center']:
                    action = "Move_Left" if free_space['left'] else ("Move_Right" if free_space['right'] else "Stop")
                    speed = self.min_speed if action != "Stop" else 0.0
                else:
                    action, speed = self.generate_vlm_command(color_image, situation_text)

                command_text = f"Action: {action}, Speed: {speed} m/s"
                print("[Command]", command_text)

                display = self.visualize(color_image, humans, command_text, situation_text)
                cv2.imshow("ASFM VLM Navigation", display)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    navigator = ASFMNavigator()
    navigator.run()
