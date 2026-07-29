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

import os
from contextlib import contextmanager
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.distributed
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, ProcessorMixin
from vllm import LLM, RequestOutput, SamplingParams

from ...protocol import DataProto
from ...utils import torch_functional as VF
from ...utils.dataset import process_image, process_video
from ...utils.torch_dtypes import PrecisionType
from .base import BaseRollout
from .config import RolloutConfig
import copy
from qwen_vl_utils import process_vision_info
from .rope_utils import get_rope_index



import json
import os
import base64
import io
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from moviepy.editor import VideoFileClip
import numpy as np
from PIL import Image
from copy import deepcopy

from .tool_video import clip_video_segment, VideoClipResult, ensure_dir

from ...utils.prompts import ROOT_SYSTEM_PROMPT, build_root_user_prompt, build_node_user_prompt

class AgentData:
    """Encapsulates all state variables for the agent loop."""

    def __init__(
        self,
        messages: list,
        node: dict,
        video_path: str,
        uid,
        node_id: str = "root",
    ):
        self.messages = messages
        self.image_data=[]
        self.id=node_id
        self.init_prompt_ids: list[int] = []
        self.uid =uid
        self.video_path=video_path

        # State variables
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.turn_scores: list[float] = []
        self.user_turns = 0
        self.assistant_turns = 0
        self.current_node = {}
        self.add_messages=[]
        self.position_ids = []
        self.input_ids = []
        self.all_mask = []
        self.all_prompt_ids = []
        self.all_response_ids = []
        self.all_response_mask = []
        self.multi_modal_inputs =[]

        # Temporary state for tool calls
        self.tool_calls: list[str] = []



def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, np.ndarray]:
    # repeat the elements, supports both tensor and numpy array
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


def _get_logit_bias(processor: Optional[ProcessorMixin]) -> Optional[dict[int, float]]:
    # enforce vllm to not output image token
    # TODO: add video token
    if processor is not None and hasattr(processor, "image_token"):
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        return {image_token_id: -100}
    else:
        return None


def _process_multi_modal_data(
    multi_modal_data: dict[str, Any], min_pixels: int, max_pixels: int, video_fps: float
) -> dict[str, Any]:
    # may convert image path to image object
    images, videos = [], []
    if "images" in multi_modal_data:
        for image in multi_modal_data["images"]:
            images.append(process_image(image, min_pixels, max_pixels))

    if "videos" in multi_modal_data:
        for video in multi_modal_data["videos"]:
            videos.append(process_video(video, min_pixels, max_pixels, video_fps))

    if len(images) != 0:
        return {"image": images}

    if len(videos) != 0:
        return {"video": videos}

    return None


class vLLMRollout(BaseRollout):
    def __init__(
        self,
        model_path: str,
        config: RolloutConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
    ):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
        """
        super().__init__()
        self.rank = int(os.getenv("RANK", "0"))
        self.config = config
        self.pad_token_id = tokenizer.pad_token_id
        self.use_tqdm = (self.rank == 0) and (not config.disable_tqdm)
        if config.tensor_parallel_size > torch.distributed.get_world_size():
            raise ValueError("Tensor parallelism size should be less than world size.")

        if config.max_num_batched_tokens < config.prompt_length + config.response_length:
            raise ValueError("max_num_batched_tokens should be greater than prompt_length + response_length.")

        engine_kwargs = {}
        if processor is not None:  # only VLMs have processor
            engine_kwargs["disable_mm_preprocessor_cache"] = True
            if config.limit_images:
                engine_kwargs["limit_mm_per_prompt"] = {"image": config.limit_images}

        self.inference_engine = LLM(
            model=model_path,
            skip_tokenizer_init=False,
            trust_remote_code=config.trust_remote_code,
            load_format="dummy",
            dtype=PrecisionType.to_str(PrecisionType.to_dtype(config.dtype)),
            seed=config.seed,
            max_model_len=config.max_model_len or config.prompt_length + config.response_length,
            distributed_executor_backend="external_launcher",
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_num_batched_tokens=config.max_num_batched_tokens,
            disable_log_stats=config.disable_log_stats,
            enforce_eager=config.enforce_eager,
            disable_custom_all_reduce=True,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_sleep_mode=True,
            **engine_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        sampling_kwargs = {
            "max_tokens": config.response_length,
            "detokenize": True,
            "logit_bias": _get_logit_bias(processor),
        }
        default_sampling_params = SamplingParams()
        for key in config.to_dict().keys():
            if hasattr(default_sampling_params, key):
                sampling_kwargs[key] = getattr(config, key)

        print(f"Sampling params: {sampling_kwargs}.")
        self.sampling_params = SamplingParams(**sampling_kwargs)
        self.frame_cache = {}
        self.tokenizer = tokenizer
        print("tokenizer type = ",type(tokenizer))
        self.system_prompt = tokenizer.apply_chat_template(
            [{}], add_generation_prompt=False, tokenize=True
        )
        self.processor = processor
        self.min_pixels=0
        self.max_pixels=0


    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)

        yield
        # roll back to previous sampling params
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences_raw(self, prompts: DataProto) -> DataProto:
        # left-padded attention_mask
        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        batch_raw_prompt_ids = non_tensor_batch.pop("raw_prompt_ids")
        batch_multi_modal_data = non_tensor_batch.pop("multi_modal_data", None)
        if batch_size != len(batch_raw_prompt_ids):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if batch_multi_modal_data is not None:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(batch_raw_prompt_ids, batch_multi_modal_data):
                vllm_inputs.append(
                    {
                        "prompt_token_ids": list(raw_prompt_ids),
                        "multi_modal_data": _process_multi_modal_data(
                            multi_modal_data,
                            prompts.meta_info["min_pixels"],
                            prompts.meta_info["max_pixels"],
                            prompts.meta_info["video_fps"],
                        ),
                    }
                )
        else:
            vllm_inputs = [{"prompt_token_ids": list(raw_prompt_ids)} for raw_prompt_ids in batch_raw_prompt_ids]

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**prompts.meta_info):
            completions: list[RequestOutput] = self.inference_engine.generate(
                prompts=vllm_inputs, sampling_params=self.sampling_params, use_tqdm=self.use_tqdm
            )
            response_ids = [output.token_ids for completion in completions for output in completion.outputs]
            response_ids = VF.pad_2d_list_to_length(
                response_ids, self.pad_token_id, max_length=self.config.response_length
            ).to(input_ids.device)

            if self.sampling_params.n > 1:
                batch_size = batch_size * self.sampling_params.n
                input_ids = _repeat_interleave(input_ids, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                if batch_multi_modal_data is not None:
                    batch_multi_modal_data = _repeat_interleave(batch_multi_modal_data, self.sampling_params.n)

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.ndim == 3:  # qwen2vl mrope: (batch_size, 4, seq_length)
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)

        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1 | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3 | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_mask = VF.get_response_mask(
            response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if batch_multi_modal_data is not None:
            non_tensor_batch = {"multi_modal_data": batch_multi_modal_data}
        else:
            non_tensor_batch = {}

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=prompts.meta_info)


    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        self.min_pixels = prompts.meta_info["min_pixels"]
        self.max_pixels = prompts.meta_info["max_pixels"]
        print("self.min_pixels,self.max_pixels  =  ",self.min_pixels,self.max_pixels)
        batch_messages = prompts.non_tensor_batch["messages"]
        print("Type of first message:", type(batch_messages[0]))

        batch_messages = [
            msg.tolist() if isinstance(msg, np.ndarray) else msg
            for msg in batch_messages
        ]
        batch_nodes = prompts.non_tensor_batch["root_node"]
        batch_video_paths = prompts.non_tensor_batch["video_path"]
        # print("video_path[0] = ",batch_video_paths[0])
        batch_uids = prompts.non_tensor_batch["uid"]
        if self.config.repeat_n > 1:
            n = self.config.repeat_n
            batch_messages = [msg for msg in batch_messages for _ in range(n)]
            batch_nodes = [node for node in batch_nodes for _ in range(n)]
            batch_video_paths = [path for path in batch_video_paths for _ in range(n)]
            batch_uids = [uid for uid in batch_uids for _ in range(n)]
        #get init node
        
        init_agent_data_list = []
        for msg,node,video_path,uid in zip(batch_messages,batch_nodes,batch_video_paths,batch_uids):
            # print("video_path = ",video_path)
            init_agent_data_list.append(AgentData(messages = copy.deepcopy(list(msg)),node=copy.deepcopy(node), video_path=copy.deepcopy(video_path), uid = uid))
        # print("init agent_data list len = ", len(init_agent_data_list))
        for agent_data in init_agent_data_list:
            prompt_text = self.processor.apply_chat_template(
                    agent_data.messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            # print("prompt_text = ",prompt_text)
            image_data,_ =process_vision_info(agent_data.messages)
            # print("len(image_data) = ",len(image_data))
            model_inputs = self.processor(text=[prompt_text],images=image_data, add_special_tokens=False,return_tensors="pt")
            temp_input_ids = model_inputs.pop("input_ids")
            agent_data.prompt_ids += temp_input_ids.squeeze(0).tolist()
            agent_data.init_prompt_ids =  temp_input_ids.squeeze(0).tolist()
            agent_data.image_data=image_data
        print("agent_data is over")        
        final_prompt_ids = []
        # final_messages_list =[]
        current_node_list =[]
        current_path_messages_list=[]
        final_agent_data_list =[]
        input_agent_data_list=copy.deepcopy(init_agent_data_list)
        
        for i in range(self.config.max_expand_depth):
            temp_final_agent_data,temp_loop_agent_data= self.get_tree_loop(input_agent_data_list)
            final_agent_data_list.extend(temp_final_agent_data)
            input_agent_data_list = temp_loop_agent_data
            if len(temp_loop_agent_data)<1:
                break
        final_agent_data_list.extend(temp_loop_agent_data)

        for agent_data in final_agent_data_list:
            
            self.tokenizer.padding_side = "left"
            prompt_output = self.tokenizer.pad(
                {"input_ids": agent_data.init_prompt_ids},
                padding="max_length",
                max_length=self.config.prompt_length,
                return_tensors="pt",
                return_attention_mask=True,
            )
            if prompt_output["input_ids"].dim() == 1:
                prompt_output["input_ids"] = prompt_output["input_ids"].unsqueeze(0)
                prompt_output["attention_mask"] = prompt_output["attention_mask"].unsqueeze(0)
            self.tokenizer.padding_side = "right"


            response_output = self.tokenizer.pad(
                {"input_ids": agent_data.response_ids},
                padding="max_length",
                max_length=self.config.response_length,
                return_tensors="pt",
                return_attention_mask=True,
            )
            if response_output["input_ids"].dim() == 1:
                response_output["input_ids"] = response_output["input_ids"].unsqueeze(0)
                response_output["attention_mask"] = response_output["attention_mask"].unsqueeze(0)

            response_mask_output = self.tokenizer.pad(
                {"input_ids": agent_data.response_mask},
                padding="max_length",
                max_length=self.config.response_length,
                return_tensors="pt",
                return_attention_mask=False,
            )
            if response_mask_output["input_ids"].dim() == 1:
                response_mask_output["input_ids"] = response_mask_output["input_ids"].unsqueeze(0)
            response_logprobs = None
            if agent_data.response_logprobs is not None:
                pad_size = self.config.response_length - len(agent_data.response_logprobs)
                response_logprobs = torch.tensor(agent_data.response_logprobs + [0.0] * pad_size).unsqueeze(0)

            response_mask = response_mask_output["input_ids"] * response_output["attention_mask"]
            attention_mask = torch.cat([prompt_output["attention_mask"], response_output["attention_mask"]], dim=1)
            input_ids = torch.cat([prompt_output["input_ids"], response_output["input_ids"]], dim=1)
            # current_text = self.tokenizer.decode(input_ids.squeeze(0))#, skip_special_tokens=True)
            
            current_text = self.processor.apply_chat_template(
                    agent_data.messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            multi_modal_inputs=self.processor(text=[current_text], images=agent_data.image_data, return_tensors="pt")
            
            multi_modal_inputs.pop("input_ids", None)
            multi_modal_inputs.pop("attention_mask", None)
            multi_modal_inputs =dict(multi_modal_inputs)
            image_grid_thw = multi_modal_inputs.get("image_grid_thw")
            

            position_ids, mrope_position_deltas = get_rope_index(
                input_ids=input_ids,  
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,  
            )
            agent_data.position_ids = position_ids
            agent_data.input_ids = input_ids
            agent_data.all_mask = attention_mask
            agent_data.all_prompt_ids = prompt_output["input_ids"]
            agent_data.all_response_ids = response_output["input_ids"]
            agent_data.all_response_mask = response_mask
            agent_data.multi_modal_inputs =multi_modal_inputs
        final_agent_data_list.sort(key=lambda x: x.uid)
        inputs = final_agent_data_list
        if len(inputs)<=0:
            print("here is a error, final data is empty")
        
        prompt_ids = torch.cat([input.all_prompt_ids for input in inputs], dim=0)
        response_ids = torch.cat([input.all_response_ids for input in inputs], dim=0)
        response_mask = torch.cat([input.all_response_mask for input in inputs], dim=0)
        attention_mask = torch.cat([input.all_mask for input in inputs], dim=0)
        input_ids = torch.cat([input.input_ids for input in inputs], dim=0)
        position_ids = torch.cat([input.position_ids for input in inputs], dim=1)

    
        # Then transpose to [batch_size, 3, seq_len] for TensorDict
        position_ids = position_ids.transpose(0, 1).contiguous()
        optional_outputs = {}
        # if inputs[0].response_logprobs is not None:
        #     optional_outputs["rollout_log_probs"] = torch.cat([input.response_logprobs for input in inputs], dim=0)
        
        print("prompt_ids size:", prompt_ids.size())
        print("response_ids size:", response_ids.size())
        print("response_mask size:", response_mask.size())
        print("input_ids size:", input_ids.size())
        print("attention_mask size:", attention_mask.size())
        print("position_ids size:", position_ids.size())
        batch = TensorDict(
            {
                "prompts": prompt_ids,  # [bsz, prompt_length]
                "responses": response_ids,  # [bsz, response_length]
                "response_mask": response_mask,  # [bsz, response_length]
                "input_ids": input_ids,  # [bsz, prompt_length + response_length]
                "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
                # position_ids: [bsz, 3, prompt_length + response_length] or [bsz, prompt_length + response_length]
                "position_ids": position_ids,
                **optional_outputs,
            },
            batch_size=len(inputs),
        )
        

        non_tensor_batch = {
            "user_turns": np.array([input.user_turns for input in inputs], dtype=np.int32),
            "assistant_turns": np.array([input.assistant_turns for input in inputs], dtype=np.int32),
            "uid": [input.uid for input in inputs],
        }

        multi_modal_inputs_list = [input.multi_modal_inputs for input in inputs]
        if any(mmi is not None for mmi in multi_modal_inputs_list):
            non_tensor_batch["multi_modal_data"] = np.array(multi_modal_inputs_list, dtype=object)
        # image_data_list = [input.image_data for input in inputs]
        # if any(mmi is not None for mmi in image_data_list):
        #     non_tensor_batch["image_data_list"] = np.array(image_data_list, dtype=object)
        
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=prompts.meta_info)
        


    def _encode_video(self, video_path: str) -> str:
        """
        Encode a video file to base64 string.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Base64 encoded string of the video
        """
        try:
            with open(video_path, "rb") as video_file:
                encoded_string = base64.b64encode(video_file.read()).decode('utf-8')
                return encoded_string
        except Exception as e:
            print(f"Warning: Failed to encode video {video_path}: {e}")
            return ""

    def _extract_initial_frames(self, video_path: str, fps: float = 10.0) -> List[Dict[str, Any]]:
        """
        Extract frames from video at specified fps and store them with timestamps.
        
        Args:
            video_path: Path to the video file
            fps: Frames per second to extract (default: 10.0)
            
        Returns:
            List of dictionaries containing base64 encoded frames and timestamps
        """
        
        if video_path in self.frame_cache:
            return self.frame_cache[video_path]
            
        frames_data = []
        with VideoFileClip(video_path) as clip:
            duration = clip.duration
            print(f"Original video duration: {duration}s")
            print(f"Original video fps: {clip.fps}")
            print(f"Original video width: {clip.w}")
            print(f"Original video height: {clip.h}")
        # import pdb; pdb.set_trace()
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
        self.frame_cache[video_path] = frames_data
        return frames_data

    def _find_nearest_frame(self, target_timestamp: float, available_frames: List[Dict[str, Any]]) -> Dict[str, Any]:
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

    def _extract_frames_with_timestamps(self, video_path: str, start_s: float = None, end_s: float = None) -> List[Dict[str, Any]]:
        """
        Extract 16 frames evenly spaced within the specified time interval,
        using the nearest available frames from the initial extraction.
        
        Args:
            video_path: Path to the video file (or time interval for tools)
            start_s: Start time in seconds (None for full video)
            end_s: End time in seconds (None for full video)
            
        Returns:
            List of dictionaries containing base64 encoded frames and timestamps
        """
        # Get all available frames
        available_frames = self.frame_cache.get(video_path, [])
        if not available_frames:
            # Fallback: extract frames if not cached
            available_frames = self._extract_initial_frames(video_path)
        
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
            nearest_frame = self._find_nearest_frame(target_timestamp, available_frames)
            
            if nearest_frame:
                # Create a copy with the target timestamp for context
                frame_copy = nearest_frame.copy()
                frame_copy["original_timestamp"] = nearest_frame["timestamp"]
                frame_copy["target_timestamp"] = target_timestamp
                target_frames.append(frame_copy)
        
        return target_frames

    def _video_meta(self, video_path: str) -> Dict[str, Any]:
        with VideoFileClip(video_path) as clip:
            return {
                "duration": float(clip.duration),
                "fps": float(clip.fps) if clip.fps else None,
                "width": int(clip.w),
                "height": int(clip.h),
                "path": video_path,
            }


    def response_2_json(self, response):
        
        import re
        match = re.search(r'<\s*TOOL_CALL\s*>(.*?)<\s*/\s*TOOL_CALL\s*>', response, re.DOTALL)
        text = match.group(1) if match else ''

        try:
            data = json.loads(text)
        except Exception:
            # Try extracting from code block
            data = None
            if "```" in text:
                parts = text.split("```")
                for i in range(len(parts)):
                    try:
                        data = json.loads(parts[i])
                        break
                    except Exception:
                        continue
            if data is None:
                data = {
                    "decision": "terminate",
                    "rationale": "JSON parse failure",
                    "proposed_paths": [],
                    "direct_answer": None,
                    "confidence": 0.0,
                }

        data.setdefault("proposed_paths", [])
        data.setdefault("direct_answer", None)
        data.setdefault("rationale", "")
        data.setdefault("decision", "terminate")
        return data
        
    def get_tree_loop(self, input_agent_data_list ):
        """
        """

        max_expand_limit = self.config.max_expand_limit 

        input_prompt_token_ids = []
        input_image_data = []
        done_agent_data=[]
        new_loop_agent_data=[]

        my_vllm_inputs = []

        debug_save_flag=0
        for agent_data in input_agent_data_list:
            
            input_prompt_token_ids.append(agent_data.prompt_ids)
            input_image_data.append(agent_data.image_data)
            
            temp_prompt_text = self.processor.apply_chat_template(
                    agent_data.messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            temp_raw_prompt_ids = self.tokenizer.encode(temp_prompt_text,add_special_tokens=False)
            my_vllm_inputs.append(
                {
                    "prompt_token_ids":list(temp_raw_prompt_ids),
                    "multi_modal_data":{"image": agent_data.image_data},
                }
            )



        completions = self.inference_engine.generate(prompts=my_vllm_inputs, sampling_params=self.sampling_params, use_tqdm=self.use_tqdm)
        
        assert len(completions)== len(input_agent_data_list)
        for i, completion in enumerate(completions):
            output = completion.outputs[0]
            
            response_text = output.text
            input_agent_data_list[i].prompt_ids += output.token_ids
            input_agent_data_list[i].response_ids += output.token_ids
            # print("token_ids:", output.token_ids)
            input_agent_data_list[i].response_mask += [1]*len(output.token_ids)
            temp_log_probs = getattr(output, 'log_probs', None)
            if temp_log_probs is not None:
                input_agent_data_list[i].response_logprobs += temp_log_probs

            # input_agent_data_list[i].response_logprobs+= output.log_probs
            input_agent_data_list[i].messages.append({"role": "assistant", "content": response_text})
            # print("response_text = ",response_text)
            info = self.response_2_json(response_text)
            # print("infro = ",info)
            nodes = {}
            decision = info.get("decision")
            if decision in ["answer", "discard"]:
                done_agent_data.append(input_agent_data_list[i])
                continue

            proposed = info.get("proposed_paths", [])[: max_expand_limit]

            if not proposed or len(proposed) == 0:

                print(f"Warning: Agent {input_agent_data_list[i].uid} has decision='{decision}' but no proposed_paths, marking as done")
                done_agent_data.append(input_agent_data_list[i])
                continue
            input_agent_data_list[i].prompt_ids += output.token_ids

            for child in proposed:
                print("here has a child")
                nodes[child.get("id")] = child
                child_agent_data =copy.deepcopy(input_agent_data_list[i])

                print(f"After deepcopy, uid = {child_agent_data.uid}")
                child_agent_data.node = child
                if child.get("tool_type") == "slide":
                    clip_res = clip_video_segment(input_agent_data_list[i].video_path, input_agent_data_list[i].node.get("start_s"), input_agent_data_list[i].node.get("end_s"), workdir=child_agent_data.node.get("workdir"), tool_type=child_agent_data.node.get("tool_type"), stride=child_agent_data.node.get("stride"))
                elif child.get("tool_type") in ["global", "local"]:
                    clip_res = clip_video_segment(input_agent_data_list[i].video_path, child_agent_data.node.get("start_s"), child_agent_data.node.get("end_s"), workdir=child_agent_data.node.get("workdir"), tool_type=child_agent_data.node.get("tool_type"))
                else:
                    done_agent_data.append(input_agent_data_list[i])
                    continue

                child_frames = self._extract_frames_with_timestamps(
                    clip_res.path, 
                    start_s=clip_res.start_s, 
                    end_s=clip_res.end_s
                )
                segment_content = []
                for frame_info in child_frames:
                    # segment_content.append({"type": "image_url", "image_url": {"url": frame_info["base64"]}})
                    segment_content.append({"type": "image_url", "image_url": frame_info['base64'], "min_pixels": self.min_pixels, "max_pixels": self.max_pixels})
                    # Show both target and original timestamps for context
                    if 'original_timestamp' in frame_info:
                        timestamp_text = f"timestamp: {frame_info['original_timestamp']:.2f}s"
                    else:
                        timestamp_text = f"timestamp: {frame_info.get('target_timestamp', frame_info['timestamp']):.2f}s"
                    segment_content.append({"type": "text", "text": timestamp_text})
                segment_content.append({"type": "text", "text": build_node_user_prompt(
                    path_id=child.get("id"),
                    strategy=child.get("strategy"),
                    start_s=child.get("start_s"),
                    end_s=child.get("end_s"),
                    clip_path=clip_res.path,
                    duration=clip_res.duration,
                    max_paths=max_expand_limit
                )})
                # child_messages = copy.copy(messages[i])
                add_messages = [
                    {
                        "role":"user",
                        "content": segment_content,
                    },
                ]
                
                prompt_text = self.processor.apply_chat_template(
                        add_messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                image_data,_ =process_vision_info(add_messages)
                model_inputs = self.processor(text=[prompt_text],images=image_data,return_tensors="pt")
                tool_call_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
                tool_call_ids = tool_call_ids[len(self.system_prompt) :]
                child_agent_data.image_data += image_data
                child_agent_data.prompt_ids += tool_call_ids
                child_agent_data.response_ids += tool_call_ids
                child_agent_data.response_mask += [0]*len(tool_call_ids)
                child_agent_data.response_logprobs += [0.0]*len(tool_call_ids)
                child_agent_data.messages.extend(add_messages)
                


                new_loop_agent_data.append(child_agent_data)
        return done_agent_data,new_loop_agent_data
