# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
from collections import defaultdict
from io import BytesIO
from typing import Any, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
from qwen_vl_utils.vision_process import fetch_video
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from . import torch_functional as VF
from .prompts import build_root_user_prompt,ROOT_SYSTEM_PROMPT

import json
import os
import base64
import io
from typing import List, Optional, Dict, Any
from moviepy.editor import VideoFileClip
import numpy as np
from PIL import Image


def find_nearest_frame(target_timestamp: float, available_frames: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Find the frame with timestamp closest to the target timestamp.
    
    Args:
        target_timestamp: The desired timestamp
        available_frames: List of available frames with timestamps
        
    Returns:
        Frame dictionary with closest timestamp
    """
    if not available_frames:
        return None
        
    closest_frame = min(available_frames, key=lambda f: abs(f["timestamp"] - target_timestamp))
    return closest_frame


def video_meta(video_path: str) -> Dict[str, Any]:
    """
    Extract metadata from video file.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dictionary containing video metadata
    """
    with VideoFileClip(video_path) as clip:
        return {
            "duration": float(clip.duration),
            "fps": float(clip.fps) if clip.fps else None,
            "width": int(clip.w),
            "height": int(clip.h),
            "path": video_path,
        }


def extract_initial_frames(video_path: str, fps: float = 10.0, frame_cache: Optional[Dict[str, List[Dict[str, Any]]]] = None) -> List[Dict[str, Any]]:
    """
    Extract frames from video at specified fps and store them with timestamps.
    
    Args:
        video_path: Path to the video file
        fps: Frames per second to extract (default: 10.0)
        frame_cache: Optional cache dictionary to store/retrieve frames
        
    Returns:
        List of dictionaries containing base64 encoded frames and timestamps
    """
    
    # Use provided cache or create a temporary one
    if frame_cache is None:
        frame_cache = {}
    
    if video_path in frame_cache:
        return frame_cache[video_path]
        
    frames_data = []
    
    with VideoFileClip(video_path) as clip:
        duration = clip.duration
        print(f"Original video duration: {duration}s")
        print(f"Original video fps: {clip.fps}")
        print(f"Original video width: {clip.w}")
        print(f"Original video height: {clip.h}")
    
    try:
        with VideoFileClip(video_path) as clip:
            duration = clip.duration
            interval = 1.0 / fps  # Time interval between frames
            
            timestamp = 0.0
            while timestamp <= duration:
                try:
                    # Extract frame at this timestamp
                    frame = clip.get_frame(timestamp)
                    
                    # Convert numpy array to PIL Image
                    pil_image = Image.fromarray(frame.astype('uint8'))
                    
                    # Convert to base64
                    buffer = io.BytesIO()
                    pil_image.save(buffer, format='JPEG')
                    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    
                    frames_data.append({
                        "base64": f"data:image/jpeg;base64,{img_base64}",
                        "timestamp": timestamp
                    })
                    
                    timestamp += interval
                    
                except Exception as e:
                    print(f"Warning: Failed to extract frame at {timestamp}s: {e}")
                    timestamp += interval
                    continue
                    
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        
    # Cache the extracted frames
    frame_cache[video_path] = frames_data
    return frames_data


def extract_frames_with_timestamps(video_path: str, start_s: float = None, end_s: float = None, 
                                 frame_cache: Optional[Dict[str, List[Dict[str, Any]]]] = None) -> List[Dict[str, Any]]:
    """
    Extract 16 frames evenly spaced within the specified time interval,
    using the nearest available frames from the initial extraction.
    
    Args:
        video_path: Path to the video file (or time interval for tools)
        start_s: Start time in seconds (None for full video)
        end_s: End time in seconds (None for full video)
        frame_cache: Optional cache dictionary containing pre-extracted frames
        
    Returns:
        List of dictionaries containing base64 encoded frames and timestamps
    """
    # Use provided cache or create a temporary one
    if frame_cache is None:
        frame_cache = {}
    
    # Get all available frames
    available_frames = frame_cache.get(video_path, [])
    if not available_frames:
        # Fallback: extract frames if not cached
        available_frames = extract_initial_frames(video_path, fps=10.0, frame_cache=frame_cache)
    
    # If no time interval specified, return all frames for initial processing
    if start_s is None or end_s is None:
        return available_frames
    
    # For tool processing: select 16 frames within the specified interval
    target_frames = []
    num_frames = 16
    
    if end_s <= start_s:
        end_s = start_s + 0.1  # Minimum interval
    
    interval_duration = end_s - start_s
    frame_interval = interval_duration / (num_frames - 1) if num_frames > 1 else 0
    
    for i in range(num_frames):
        target_timestamp = start_s + (i * frame_interval)
        nearest_frame = find_nearest_frame(target_timestamp, available_frames)
        
        if nearest_frame:
            # Create a copy with the target timestamp for context
            frame_copy = nearest_frame.copy()
            frame_copy["original_timestamp"] = nearest_frame["timestamp"]
            frame_copy["target_timestamp"] = target_timestamp
            target_frames.append(frame_copy)
    
    return target_frames

def extract_frames_from_file(video_path: str, start_s: float = None, end_s: float = None, 
                                 available_frames: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Extract 16 frames evenly spaced within the specified time interval,
    using the nearest available frames from the initial extraction.
    
    Args:
        video_path: Path to the video file (or time interval for tools)
        start_s: Start time in seconds (None for full video)
        end_s: End time in seconds (None for full video)
        frame_cache: Optional cache dictionary containing pre-extracted frames
        
    Returns:
        List of dictionaries containing base64 encoded frames and timestamps
    """


    
    # If no time interval specified, return all frames for initial processing
    if start_s is None or end_s is None:
        return available_frames
    
    # For tool processing: select 16 frames within the specified interval
    target_frames = []
    num_frames = 16
    
    if end_s <= start_s:
        end_s = start_s + 0.1  # Minimum interval
    
    interval_duration = end_s - start_s
    frame_interval = interval_duration / (num_frames - 1) if num_frames > 1 else 0
    
    for i in range(num_frames):
        target_timestamp = start_s + (i * frame_interval)
        nearest_frame = find_nearest_frame(target_timestamp, available_frames)
        
        if nearest_frame:
            # Create a copy with the target timestamp for context
            frame_copy = nearest_frame.copy()
            frame_copy["original_timestamp"] = nearest_frame["timestamp"]
            frame_copy["target_timestamp"] = target_timestamp
            target_frames.append(frame_copy)
    
    return target_frames


def collate_fn(features: list[dict[str, Any]]) -> dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)
        non_tensors["root_node"].append(
            {
                "id":"root",
                "start_s":0,
                "end_s":video_meta(feature.get("video_path",""))["duration"],
            }
        )


    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


def process_image(
    image: Union[dict[str, Any], ImageObject, str], min_pixels: Optional[int], max_pixels: Optional[int]
) -> ImageObject:
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))

    image.load()  # avoid "Too many open files" errors
    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if min_pixels is not None and (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def process_video(
    video: str, min_pixels: Optional[int], max_pixels: Optional[int], video_fps: float, return_fps: bool = False
) -> Union[list[ImageObject], tuple[list[ImageObject], list[float]]]:
    vision_info = {"video": video, "min_pixels": min_pixels, "max_pixels": max_pixels, "fps": video_fps}
    return fetch_video(vision_info, return_video_sample_fps=return_fps)


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        video_key: str = "videos",
        image_dir: Optional[str] = None,
        video_fps: float = 2.0,
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,
        filter_overlong_prompts_workers: int = 16,
        max_expand_limit: int = 3,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.video_key = video_key
        self.image_dir = image_dir
        self.video_fps = video_fps
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.frame_cache={}
        self.max_expand_limit=max_expand_limit

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isdir(data_path):
            self.dataset = load_dataset(data_path, split=data_split)
            # when we use dataset builder, we should always refer to the train split
            # file_type = os.path.splitext(os.listdir(data_path)[0])[-1][1:].replace("jsonl", "json")
            # self.dataset = load_dataset(file_type, data_dir=data_path, split=data_split)
        elif os.path.isfile(data_path):
            file_type = os.path.splitext(data_path)[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_files=data_path, split=data_split)
        else:
            # load remote dataset from huggingface hub
            self.dataset = load_dataset(data_path, split=data_split)

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        if filter_overlong_prompts:
            self.dataset = self.dataset.filter(
                self._filter_overlong_prompts,
                desc="Filtering overlong prompts",
                num_proc=filter_overlong_prompts_workers,
            )

    def _build_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        prompt_str: str = example[self.prompt_key]
        if "video_path" in example:
            if "messages" in example:
                return example["messages"]
            video_path = example["video_path"]
            frame_cache_path = example["frame_cache_path"]
            
            video_dir = os.path.dirname(video_path)
            video_filename = os.path.basename(video_path)
            video_name = os.path.splitext(video_filename)[0]
            workdir = os.path.join(video_dir, f"{video_name}_work")
            os.makedirs(workdir, exist_ok=True)
            meta = video_meta(video_path)

            frame_meta = {}
            with open(frame_cache_path, 'r', encoding='utf-8') as f:
                frame_meta = json.load(f)
            frames_data =  frame_meta["frames"]



            segment_frames = extract_frames_from_file(
                video_path, 
                start_s=0, 
                end_s=meta["duration"], 
                available_frames=frames_data,
            )
            frame_content=[]
            question =example["full_question"]
            for frame_info in segment_frames:
                frame_content.append({"type": "image_url", "image_url": frame_info['base64'], "min_pixels": self.min_pixels, "max_pixels": self.max_pixels})
                frame_content.append({"type": "text", "text": f"timestamp: {frame_info['timestamp']}s"})

            frame_content.append({"type": "text", "text": build_root_user_prompt(meta, question, max_paths=self.max_expand_limit)})

            messages = [
                {"role": "system", "content": ROOT_SYSTEM_PROMPT},
                {"role": "user", "content": frame_content},
            ]
            example["messages"]=messages
            return  messages

        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            prompt_str = format_prompt.render(content=prompt_str)

        if self.image_key in example:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []
            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        elif self.video_key in example:
            content_list = []
            for i, content in enumerate(prompt_str.split("<video>")):
                if i != 0:
                    content_list.append({"type": "video"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        else:
            return [{"role": "user", "content": prompt_str}]

    def _filter_overlong_prompts(self, example: dict[str, Any]) -> bool:
        messages = self._build_messages(example)
        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example[self.image_key]
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        elif self.video_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example[self.video_key]
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            for video in videos:
                processed_videos.append(process_video(video, self.min_pixels, self.max_pixels, self.video_fps))

            model_inputs = self.processor(
                videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        else:
            input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            return len(input_ids) <= self.max_prompt_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example: dict = self.dataset[index]
        messages = self._build_messages(example)
        example.pop(self.prompt_key, None)

        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example.pop(self.image_key)
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"images": images}
        elif self.video_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example.pop(self.video_key)
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            video_fps_list = []
            for video in videos:
                processed_video, video_fps = process_video(
                    video, self.min_pixels, self.max_pixels, self.video_fps, return_fps=True
                )
                processed_videos.append(processed_video)
                video_fps_list.append(video_fps)

            model_inputs = self.processor(
                videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            if "second_per_grid_ts" in self.processor.model_input_names:
                model_inputs["second_per_grid_ts"] = [2.0 / video_sample_fps for video_sample_fps in video_fps_list]

            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"videos": videos}
        else:
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen-vl mrope
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from ..models.transformers.qwen3_vl import get_rope_index
            else:
                from ..models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                video_grid_thw=model_inputs.get("video_grid_thw", None),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                attention_mask=attention_mask,
            )  # (3, seq_length)
            text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)  # (1, seq_length)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)  # (4, seq_length)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        example["raw_prompt_ids"] = raw_prompt_ids
        example["ground_truth"] = example.pop(self.answer_key)
        return example
