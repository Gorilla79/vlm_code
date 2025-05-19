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
        self.vlm_model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b").to('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.A = 2.1
        self.B = 0.3
        self.C = 0.0
        self.personal_space = 1.2
        self.min_speed = 0.2
        self.max_speed = 1.5
        self.last_human_positions = {}

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
            if int(box.cls[0]) == 0:
                bbox = box.xyxy[0].cpu().numpy()
                cx = int((bbox[0] + bbox[2]) / 2)
                cy = int((bbox[1] + bbox[3]) / 2)
                pos_3d = self.deproject(depth_frame, (cx, cy))
                if pos_3d is not None and pos_3d[2] > 0:
                    humans.append({'position': pos_3d, 'bbox': bbox})
        return humans

    def update_human_velocities(self, humans):
        current_time = time.time()
        for human in humans:
            pos = human['position']
            human_id = f"{pos[0]:.2f}_{pos[1]:.2f}_{pos[2]:.2f}"
            last = self.last_human_positions.get(human_id)
            self.last_human_positions[human_id] = (pos, current_time)
            if last:
                last_pos, last_time = last
                dt = current_time - last_time
                if dt > 0.05:
                    human['velocity'] = (pos - last_pos) / dt
                else:
                    human['velocity'] = np.zeros(3)
            else:
                human['velocity'] = np.zeros(3)

    def compute_asfm_force(self, robot_pos, humans):
        total_force = np.zeros(3)
        for human in humans:
            diff = robot_pos - human['position']
            dist = np.linalg.norm(diff)
            if dist < self.personal_space:
                static_force = self.A * np.exp(self.B * dist + self.C)
                moving_force = self.A * np.exp((dist - 0.5) * self.B + self.C)
                direction = diff / (dist + 1e-6)
                force_vec = (static_force + moving_force) * direction
                total_force += force_vec
        return total_force

    def generate_vlm_command(self, color_frame, humans, asfm_direction, asfm_speed):
        prompt = (
            f"Current situation: {len(humans)} humans detected.\n"
            f"Human positions: {[list(h['position']) for h in humans]}\n"
            f"ASFM Suggested Direction: {asfm_direction}, Speed: {asfm_speed:.2f} m/s.\n"
            "What is the safest direction (Left, Right, Forward, Stop) and optimal speed (0.2-1.5 m/s)?"
        )
        inputs = self.processor(images=Image.fromarray(cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)),
                                 text=prompt, return_tensors="pt").to(self.device)
        outputs = self.vlm_model.generate(**inputs, max_new_tokens=50)
        decoded = self.processor.decode(outputs[0], skip_special_tokens=True)
        match = re.search(r"(Left|Right|Forward|Stop).*?(\d+\.\d+)", decoded)
        if match:
            action = match.group(1)
            speed = float(match.group(2))
            return action, speed
        return asfm_direction, asfm_speed

    def run(self):
        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())

                humans = self.detect_humans(color_image, depth_frame)
                self.update_human_velocities(humans)

                force = self.compute_asfm_force(np.array([0, 0, 0]), humans)
                force_magnitude = np.linalg.norm(force)
                if force_magnitude > 1e-3:
                    asfm_direction_vector = force / (force_magnitude + 1e-6)
                else:
                    asfm_direction_vector = np.array([0, 0, 1])

                # ASFM Direction Text
                if abs(asfm_direction_vector[0]) > abs(asfm_direction_vector[2]):
                    asfm_direction = "Left" if asfm_direction_vector[0] < 0 else "Right"
                else:
                    asfm_direction = "Forward"

                asfm_speed = np.clip(force_magnitude, self.min_speed, self.max_speed)

                # VLM decision priority based on number of humans
                if len(humans) >= 3:
                    direction, speed = self.generate_vlm_command(color_image, humans, asfm_direction, asfm_speed)
                else:
                    direction, speed = asfm_direction, asfm_speed

                display = color_image.copy()
                for human in humans:
                    x1, y1, x2, y2 = map(int, human['bbox'])
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

                info_text = f"Direction: {direction}, Speed: {speed:.2f} m/s"
                cv2.putText(display, info_text, (10, display.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                cv2.imshow("ASFM + VLM Navigation", display)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    navigator = ASFMNavigator()
    navigator.run()
