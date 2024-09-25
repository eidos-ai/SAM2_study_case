import cv2
import json
import numpy as np
import argparse
from pathlib import Path
import threading

class VideoShow:
    def __init__(self, video_path, annotations_path, annotation_name):
        # Open existing annotations
        if Path(annotations_path).exists():
            with open(annotations_path) as f:
                self.annotations = json.load(f)
        else:
            print(annotation_name)
            self.annotations = {
                "player_name": annotation_name,
                "player_track": []
            }
        self.annotations_path = annotations_path
        self.annotation_name = annotation_name
        # Open the video file
        self.cap = cv2.VideoCapture(video_path)
        self.video_length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.tag_point_list = {frame_i: None for frame_i in range(1, self.video_length + 1)}
        self.move_next = True
        self.keep_moving = False
        self.frame_i = -1
        self.tag_position = None
        self.tagging = True
        self.press_moving_control = False

        self.timeline_controller = TimelineController(width=500, height=100, video_player = self)

        on_press = {
            frozenset(['s']) : self.jump_start,
            frozenset([Key.space]) : self.play_pause,
            frozenset([Key.ctrl]): self.press_moving_control,
            frozenset(['n']): self.prev_frame,
            frozenset(['m']): self.next_frame,
        }
        on_release = {
            frozenset([Key.ctrl]): self.press_moving_control
        }
        self.key_listener = KeyListener(on_press)

    def press_moving_control(self):
        self.press_moving_control = not self.press_moving_control

    def jump_start(self):
        self.frame_i = -1
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def play_pause(self):
        self.keep_moving = not self.keep_moving
        self.move_next = not self.move_next

    def prev_frame(self):
        self.frame_i = max(0, self.frame_i - 1)
            
    def next_frame(self):
        self.frame_i = min(self.video_length - 1, self.frame_i + 1)

    # Define a callback function to handle mouse events
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.tag_position = (x, y)
            self.tag_point_list[self.frame_i + 1] = self.tag_position
            print(self.tag_position)

    def render_point(self, point):
        cv2.circle(self.render_frame, point, 10, (0, 0, 255), -1)

    def write_text(self, text, x, y):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (0, 0, 255) # in BGR format
        thickness = 2
        cv2.putText(self.render_frame, text, (x, y), font, font_scale, color, thickness)

    def jump(self, frame_i):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_i)
        self.frame_i = frame_i
        _, self.frame = self.cap.read()

    def run(self):
        # Create a window to display the frames
        cv2.namedWindow('frame')
        cv2.setMouseCallback('frame', self.mouse_callback)
        while True:
            if self.move_next or self.frame_i == -1:
                # Tag current Frame
                if self.tagging and self.tag_position is not None:
                    self.tag_point_list[self.frame_i + 1] = self.tag_position
                ret, self.frame = self.cap.read()
                self.frame_i += 1
                # If the frame could not be read, break out of the loop
                if not ret:
                    break
                if not self.keep_moving:
                    self.move_next = False

            # Render on frame
            self.render_frame = self.frame.copy()
            self.write_text(f"frame {self.frame_i + 1}", 100, 100)
            text_mode = "TAGGING" if self.tagging else "SHOW"
            self.write_text(f"{text_mode}", 100, 200)
            self.render_point(self.tag_point_list[self.frame_i + 1])
            # Display the frame
            cv2.imshow('frame', self.render_frame)

            self.timeline_controller.draw()

            # Wait for the user to click the mouse
            key = cv2.waitKey(1)
            if key & 0xFF == ord('s'):
                self.frame_i = -1
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if key & 0xFF == ord(' '):
                self.keep_moving = not self.keep_moving
                self.move_next = not self.move_next
            if key & 0xFF == ord('n'):
                self.frame_i = max(0, self.frame_i - 1)
                self.jump(self.frame_i)
            if key & 0xFF == ord('m'):
                self.frame_i = min(self.video_length - 1, self.frame_i + 1)
                self.jump(self.frame_i)

            # Exit command
            if key & 0xFF == 27:
                break
            if key & 0xFF == ord('q'):
                break

        # Release the video capture and close the window
        self.cap.release()
        cv2.destroyAllWindows()
        new_annotation = {
                "player_name": self.annotation_name,
                "player_track": [
            [frame, point[0], point[1]] for frame, point in self.tag_point_list.items() if point is not None
            ]
        }
        self.annotations.append(new_annotation)

        # Write the dictionary to a JSON file
        with open(self.annotations_path, 'w') as f:
            json.dump(self.annotations, f, indent=4)

class TimelineController():
    WIDTH_PLAY_BAR = 5

    def __init__(self, width, height, video_player):
        self.width = width
        self.height = height
        self.video_player = video_player
        self.timeline = np.zeros((self.height, self.width, 1), dtype=np.uint8)
        self.timeline.fill(16)

        cv2.namedWindow('control')
        cv2.setMouseCallback('control', self.mouse_callback)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_position = x
            current_relative_position = self.current_position / self.width
            clicked_frame = int(current_relative_position * self.video_player.video_length)
            self.video_player.jump(clicked_frame)

    def draw(self):
        current_relative_position = self.video_player.frame_i / self.video_player.video_length
        self.render_timeline = self.timeline.copy()
        current_position = int(self.width * current_relative_position)

        # Render current time position
        cv2.rectangle(self.render_timeline, (current_position, 0), (current_position+TimelineController.WIDTH_PLAY_BAR, self.height), color=(255, 255, 255), thickness=-1)
        cv2.imshow('control', self.render_timeline )

from pynput.keyboard import Key, Listener

class KeyListener:
    def __init__(self, combination_to_function):
        self.current_keys = set()
        self.combination_to_function = combination_to_function

    def on_press(self, key):
        # When a key is pressed, add it to the set we are keeping track of and check if this set is in the dictionary
        self.current_keys.add(key)
        if frozenset(self.current_keys) in self.combination_to_function:
            # If the current set of keys are in the mapping, execute the function
            self.combination_to_function[frozenset(self.current_keys)]()

    def on_release(self, key):
        # When a key is released, remove it from the set of keys we are keeping track of
        if key in self.current_keys:
            self.current_keys.remove(key)

    def init(self):
        listener = Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()

def tag_video(video_path, annotation_name, annotations_path):
    video_show = VideoShow(video_path, annotations_path, annotation_name)
    video_show.run()

# video_show = VideoShow("/Users/ortegatron/eidos/video drone/drone.mp4", "test.json",annotation_name)
# video_show.run()

def main():
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Process video and annotations data")
    # Add the video argument
    parser.add_argument("--video", type=str, required=True, help="path to the video file")
    # Add the annotations argument
    parser.add_argument("--name", type=str, required=True, help="name of the annotations to be tagged")
    # Add the annotations argument
    parser.add_argument("--annotations", type=str, required=True, help="path to the annotations file")
    # Parse the arguments
    args = parser.parse_args()
    # Call the process_data function with the arguments
    tag_video(args.video, args.name, args.annotations)

if __name__ == '__main__':
    main()
