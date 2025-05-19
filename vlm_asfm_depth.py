import pyrealsense2 as rs
import numpy as np
import cv2
import time
import torch
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import re

class ASFMNavigator:
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(config)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.yolo = YOLO('yolov8n.pt').to(self.device)

        # Jetson에 사전 다운로드된 flan-t5-xl 모델 사용
        model_path = "/home/unitree/.cache/huggingface/hub/models--Salesforce--blip2-flan-t5-base/snapshots/<revision>"
        self.processor = BlipProcessor.from_pretrained(model_path, local_files_only=True)
        self.vlm_model = BlipForConditionalGeneration.from_pretrained(model_path, local_files_only=True).to(self.device)

        self.A = 2.1
        self.B = 0.3
        self.C = 0.0
        self.side_step_lambda_thres = 2.0
        self.follow_angle_thres = np.deg2rad(30)
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
                human['velocity'] = (pos - last_pos) / dt if dt > 0.05 else np.zeros(3)
            else:
                human['velocity'] = np.zeros(3)

    def compute_adaptive_speed(self, humans):
        if not humans:
            return self.max_speed
        min_speed = self.max_speed
        for human in humans:
            dist = np.linalg.norm(human['position'])
            vel = human.get('velocity', np.zeros(3))
            approaching = np.dot(vel, human['position']) < 0

            if dist < 0.3:
                speed = 0.0
            elif dist < 2.0:
                ratio = (dist - 0.3) / (2.0 - 0.3)
                speed = self.min_speed + ratio * (self.max_speed - self.min_speed)
                if approaching:
                    speed *= 0.7
            else:
                speed = self.max_speed
            min_speed = min(min_speed, speed)
        return min_speed

    def compute_asfm_force(self, robot_pos, humans):
        total_force = np.zeros(3)
        for human in humans:
            diff = robot_pos - human['position']
            dist = np.linalg.norm(diff)
            if dist < self.personal_space:
                static_force = self.A * np.exp(self.B * dist + self.C)
                moving_force = self.A * np.exp((dist - 0.5) * self.B + self.C)
                direction = diff / (dist + 1e-6)
                total_force += (static_force + moving_force) * direction
        return total_force

    def analyze_environment(self, depth_image, humans):
        h, w = depth_image.shape
        left = np.mean(depth_image[:, :w//3])
        center = np.mean(depth_image[:, w//3:2*w//3])
        right = np.mean(depth_image[:, 2*w//3:])

        free_space = {
            'left': left > 1.0,
            'center': center > 1.0,
            'right': right > 1.0
        }

        approaching = any(np.dot(h['velocity'], h['position']) < 0 for h in humans)
        moving_away = any(np.dot(h['velocity'], h['position']) > 0 for h in humans)

        text = f"Free space - Left: {'Yes' if free_space['left'] else 'No'}, "
        text += f"Center: {'Yes' if free_space['center'] else 'No'}, "
        text += f"Right: {'Yes' if free_space['right'] else 'No'}. "
        if approaching:
            text += "Some humans are approaching. "
        if moving_away:
            text += "Some humans are moving away. "
        if not humans:
            text += "No humans detected. "

        return text, free_space

    def generate_vlm_command(self, color_frame, situation_text):
        prompt = (
            f"Situation: {situation_text} "
            f"Suggest the best movement action and safe speed (0.2-1.5m/s). "
            f"Actions: Move_Left, Move_Right, Move_Straight, Stop"
        )
        inputs = self.processor(
            images=Image.fromarray(cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)),
            text=prompt, return_tensors="pt"
        ).to(self.device)
        outputs = self.vlm_model.generate(**inputs, max_new_tokens=50)
        decoded = self.processor.decode(outputs[0], skip_special_tokens=True)
        match = re.search(r"(Move_Left|Move_Right|Move_Straight|Stop)\s+(\d+(\.\d+)?)m/s", decoded)
        if match:
            return match.group(1), float(match.group(2))
        return "Move_Straight", 1.0

    def visualize(self, frame, humans, command, situation_text):
        for human in humans:
            x1, y1, x2, y2 = map(int, human['bbox'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, situation_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, f"Command: {command}", (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
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
                depth_image = np.asanyarray(depth_frame.get_data())

                humans = self.detect_humans(color_image, depth_frame)
                self.update_human_velocities(humans)

                situation_text, free_space = self.analyze_environment(depth_image, humans)
                print("[Situation]", situation_text)

                if len(humans) >= 3:
                    action, speed = self.generate_vlm_command(color_image, situation_text)
                    direction = {
                        "Move_Left": np.array([-1, 0, 0]),
                        "Move_Right": np.array([1, 0, 0]),
                        "Move_Straight": np.array([0, 0, 1]),
                        "Stop": np.array([0, 0, 0])
                    }.get(action, np.array([0, 0, 1]))
                else:
                    force = self.compute_asfm_force(np.array([0, 0, 0]), humans)
                    direction = force / (np.linalg.norm(force) + 1e-6) if np.linalg.norm(force) > 1e-3 else np.array([0, 0, 1])
                    speed = self.compute_adaptive_speed(humans)

                command_text = f"Move ({direction[0]:.2f}, {direction[1]:.2f}, {direction[2]:.2f}) at {speed:.2f}m/s"
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
