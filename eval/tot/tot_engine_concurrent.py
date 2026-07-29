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
import random
import pandas as pd
from tool_video import clip_video_segment, VideoClipResult, ensure_dir
from prompts import ROOT_SYSTEM_PROMPT, build_root_user_prompt, build_node_user_prompt

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy data types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


@dataclass
class PathNode:
    path_id: str
    start_s: float
    end_s: float
    strategy: str
    depth: int
    parent_id: Optional[str] = None
    tool_type: str = "global"  # global| local | slide
    stride: float = 0.0  # Only for "slide" type
    father_start_s: float = 0.0
    father_end_s: float = 0.0
    clip_result: Optional[VideoClipResult] = None
    status: str = "pending"  # pending | processed | discarded
    decision: Optional[str] = None
    direct_answer: Optional[str] = None
    rationale: Optional[str] = None
    confidence: Optional[float] = None
    children: List[str] = field(default_factory=list)
    messages: Optional[List[Dict[str, str]]] = None

@dataclass
class ToTRunResult:
    root_paths: List[str]
    nodes: Dict[str, PathNode]
    final_answer: Optional[str] = None
    terminated: bool = False
    all_answers: List[Dict[str, Any]] = field(default_factory=list)
    confidence: Optional[float] = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    api_call_count: int = 0

class ConcurrentToTEngine:
    """线程安全的ToT引擎，每个实例都有独立的状态"""
    
    def __init__(
        self,
        llm_model: str = "gpt-4o-mini",
        base_url: str = "http://localhost:9010/v1",
        api_key: Optional[str] = None,
        max_depth: int = 3,
        per_expand_limit: int = 3,
        temperature: float = 0.2,
        thread_safe: bool = True,
        early_stop: bool = True,  # 新增参数：是否在找到答案时提前停止
    ):
        self.llm_model = llm_model
        self.max_depth = max_depth
        self.per_expand_limit = per_expand_limit
        self.temperature = temperature
        self.thread_safe = thread_safe
        self.base_url = base_url
        self.early_stop = early_stop  # 控制是否提前停止
        self.thread_id = threading.get_ident() if thread_safe else None
        
        # 每个实例独立的客户端
        self.client = None
        self.llm_model = llm_model
        if OpenAI is not None:
            self.client = OpenAI(
                api_key="",
                base_url=self.base_url,
            )

        # 实例独立的状态 - 每次run都会重置
        self.messages: List[Dict[str, str]] = []
        self.frame_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.workdir: Optional[str] = None
        
        # Token统计
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_tokens: int = 0
        self.api_call_count: int = 0
        
        # 线程安全锁（如果需要）
        if thread_safe:
            self._lock = threading.RLock()
        else:
            self._lock = None
        self.cache_max_size = 8#可调整，frame cache的最大容量

    def _thread_safe_operation(self, func, *args, **kwargs):
        """线程安全操作包装器"""
        if self._lock:
            with self._lock:
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    def _cleanup_by_lru(self):
        """基于 LRU 策略清理缓存"""
        while len(self.frame_cache) > self.cache_max_size:
            self.frame_cache.popitem(last=False)  # 移除最老的项
    
    def _extract_initial_frames(self, video_path: str, fps: float = 10.0) -> List[Dict[str, Any]]:
        """
        Extract frames from video at specified fps and store them with timestamps.
        线程安全版本 - 使用实例级缓存
        """
        # 检查实例级缓存
        if video_path in self.frame_cache:
            return self.frame_cache[video_path]
        keywords = ['egoschema', 'mlvu', 'tomato', 'videomme', 'vsibench', 'longvideobench']
        
        frame_cache_path = ""
        for keyword in keywords:
            if f'{keyword}' in video_path.lower().replace('-', ''):
                # 提取文件名
                basename = os.path.basename(video_path)
                filename, extension = os.path.splitext(basename)
                # 构造输出路径
                frame_cache_path =  f"/data/custom_frame_cache_resize/{keyword}/{filename}_fps8.0.json"
                break
        if frame_cache_path !="":
            with open(frame_cache_path, 'r', encoding='utf-8') as file:
                frames_cache = json.load(file)
            self.frame_cache[video_path] = frames_cache["frames"]
            return frames_cache["frames"]

            
        frames_data = []

        from moviepy.editor import VideoFileClip
        from moviepy.video.fx.resize import resize
        
        try:
            print("use moviepy to extract frames")
            with VideoFileClip(video_path) as clip_origin:
                # 先缩小整个视频 clip，再抽帧
                clip = resize(clip_origin, 0.75)  # 0.75 表示 75% 尺寸
                duration = clip.duration
                print(f"[Thread {self.thread_id}] Video duration: {duration}s, fps: {clip.fps}")
                
                interval = 1.0 / fps
                timestamp = 0.0
                
                while timestamp <= duration:
                    try:
                        frame = clip.get_frame(timestamp)
                        pil_image = Image.fromarray(frame.astype('uint8'))

                        buffer = io.BytesIO()
                        pil_image.save(buffer, format='JPEG')   # quality 保持默认
                        
                        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                        
                        frames_data.append({
                            "base64": img_base64,
                            "timestamp": timestamp
                        })
                        
                        timestamp += interval
                        
                    except Exception as e:
                        print(f"Warning: Failed to extract frame at {timestamp}s: {e}")
                        timestamp += interval
                        continue
                        
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            
        # 存储到实例级缓存
        self.frame_cache[video_path] = frames_data
        return frames_data

    
    def _find_nearest_frame(self, target_timestamp: float, available_frames: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find the frame with timestamp closest to the target timestamp."""
        if not available_frames:
            return None
            
        closest_frame = min(available_frames, key=lambda f: abs(f["timestamp"] - target_timestamp))
        return closest_frame

    def _extract_frames_with_timestamps(self, video_path: str, start_s: float = None, end_s: float = None, max_frames: int = 16, fps: float = 8.0) -> List[Dict[str, Any]]:
        """
        Extract 16 frames evenly spaced within the specified time interval.
        """
        # 获取可用帧
        available_frames = self.frame_cache.get(video_path, [])
        if not available_frames:
            available_frames = self._extract_initial_frames(video_path, fps=fps)
        
        if start_s is None or end_s is None:
            return available_frames
        
        # 选择16帧
        target_frames = []
        num_frames = max_frames

        if end_s <= start_s:
            end_s = start_s + 0.1
        
        interval_duration = end_s - start_s
        frame_interval = interval_duration / (num_frames - 1) if num_frames > 1 else 0
        
        for i in range(num_frames):
            target_timestamp = start_s + (i * frame_interval)
            nearest_frame = self._find_nearest_frame(target_timestamp, available_frames)
            
            if nearest_frame:
                frame_copy = nearest_frame.copy()
                frame_copy["original_timestamp"] = nearest_frame["timestamp"]
                frame_copy["target_timestamp"] = target_timestamp
                target_frames.append(frame_copy)
        
        return target_frames

    def _chat_to_json(self, messages: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Send messages to the LLM and parse a strict JSON response.
        线程安全版本 - 使用独立的消息副本
        """
        use_global = messages is None
        if use_global:
            msgs = deepcopy(self.messages)  # 使用深拷贝避免状态污染
        else:
            msgs = deepcopy(messages)

        if self.client is None:
            # Offline fallback
            dummy = {
                "decision": "expand",
                "rationale": "Offline fallback: propose probing paths.",
                "proposed_paths": [
                    {"id": "P1", "strategy": "probe beginning", "start_s": 0, "end_s": 5},
                    {"id": "P2", "strategy": "probe middle", "start_s": 5, "end_s": 10},
                ],
                "direct_answer": None,
                "evidence_confidence": 0.3, 
            }
            
            # 更新相应的消息列表
            if use_global:
                self.messages.append({"role": "assistant", "content": json.dumps(dummy, ensure_ascii=False)})
            else:
                messages.append({"role": "assistant", "content": json.dumps(dummy, ensure_ascii=False)})
            return dummy

        try:
            # Retry logic: if resp is None, retry up to 10 times
            resp = None
            for attempt in range(4):
                try:
                    # Check the structure of the last message and directly modify msgs
                    if isinstance(msgs[-1]["content"], list):
                        # Find the last item with type "text"
                        for item in reversed(msgs[-1]["content"]):
                            if item.get("type") == "text" and "text" in item:
                                item["text"] = item["text"] + "\n/no_think\n"
                                break
                        # print(msgs[-1])
                    else:
                        # Content is a string
                        msgs[-1]["content"] = msgs[-1]["content"] + "\n/no_think\n"
                    
                    resp = self.client.chat.completions.create(
                        model=self.llm_model,
                        temperature=0.1,
                        messages=msgs,
                    )
                    # print(resp)
                    if resp is not None:
                        break
                except Exception as e:
                    print(f"[Thread {self.thread_id}] API call attempt {attempt + 1} failed: {e}")
                    if attempt == 3:  # Last attempt
                        raise e
            
            if resp is None:
                raise Exception("API call failed after 10 attempts")
                
            text = resp.choices[0].message.content.strip()
            origin_text = text
            
            # 收集token使用信息
            if hasattr(resp, 'usage') and resp.usage is not None:
                self.api_call_count += 1
                if hasattr(resp.usage, 'prompt_tokens'):
                    self.total_input_tokens += resp.usage.prompt_tokens
                if hasattr(resp.usage, 'completion_tokens'):
                    self.total_output_tokens += resp.usage.completion_tokens
                if hasattr(resp.usage, 'total_tokens'):
                    self.total_tokens += resp.usage.total_tokens
                else:
                    self.total_tokens = self.total_input_tokens + self.total_output_tokens

            # 更新相应的消息列表
            if use_global:
                self.messages.append({"role": "assistant", "content": text})
            else:
                messages.append({"role": "assistant", "content": text})
            
            # 解析JSON
            raw_text = text
            # print(text)
            # import re
            # match = re.search(r'<\s*TOOL_CALL\s*>(.*?)<\s*/\s*TOOL_CALL\s*>', raw_text, re.DOTALL)
            # text = match.group(1) if match else ''
            # print("-=====-")
            # print(text)
            import re

            # 先尝试匹配 TOOL_CALL
            match = re.search(r'<\s*TOOL_CALL\s*>(.*?)<\s*/\s*TOOL_CALL\s*>', raw_text, re.DOTALL)

            # 如果没匹配到，再尝试匹配 TOOLCALL
            if not match:
                # print("-0000-")
                # print(raw_text)
                match = re.search(r'<\s*TOOLCALL\s*>(.*?)<\s*/\s*TOOLCALL\s*>', raw_text, re.DOTALL)

            text = match.group(1) if match else ''

            # print("-0000-")
            # print(text)

            if text == '':
                m = re.search(
                    r'(?:^|\n)\s*.*?[:：]?\s*```(?:json)?\n(.*?)\n\s*```',
                    raw_text, re.DOTALL | re.I
                )
                text = m.group(1).strip() if m else ''

            try:
                data = json.loads(text)
            except Exception:
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
                        "evidence_confidence": 0.0,
                    }

            data.setdefault("proposed_paths", [])
            # 检查direct_answer键，如果不存在则尝试从origin_text中提取\boxed{}内容
            if "direct_answer" not in data:
                import re
                boxed_match = re.search(r'\\boxed\{([^}]*)\}', origin_text)
                if boxed_match:
                    data["direct_answer"] = boxed_match.group(1)
                    data["decision"] = "answer"
                else:
                    data["direct_answer"] = None
            data.setdefault("rationale", "")
            data.setdefault("decision", "terminate")
            # print("=====")
            # print(data)
            return data
            
        except Exception as e:
            print(f"[Thread {self.thread_id}] Error in LLM call: {e}")
            return {
                "decision": "terminate",
                "rationale": f"LLM call failed: {str(e)}",
                "proposed_paths": [],
                "direct_answer": None,
                "evidence_confidence": 0.0,
            }

    def _video_meta(self, video_path: str) -> Dict[str, Any]:
        """获取视频元数据"""
        with VideoFileClip(video_path) as clip:
            return {
                "duration": float(clip.duration),
                "fps": float(clip.fps) if clip.fps else None,
                "width": int(clip.w),
                "height": int(clip.h),
                "path": video_path,
            }

    def _sanitize_paths(self, proposed: List[Dict[str, Any]], parent_id: Optional[str], duration: float, depth: int) -> List[PathNode]:
        """清理和验证提议的路径"""
        nodes: List[PathNode] = []
        for i, p in enumerate(proposed):
            pid = str(p.get("id") or f"P{depth}_{i+1}")
            strat = str(p.get("strategy") or "expand")
            start_s = float(p.get("start_s", 0.0))
            end_s = float(p.get("end_s", max(0.1, min(duration, start_s + 5))))
            tool_type = str(p.get("tool_type", "global")).lower()
            stride = float(p.get("stride", 0.0))
            
            if start_s < 0:
                start_s = 0.0
            if end_s <= start_s:
                end_s = min(duration, start_s + 2.0)
            end_s = min(end_s, duration)
            
            node = PathNode(
                path_id=pid,
                start_s=start_s,
                end_s=end_s,
                strategy=strat,
                depth=depth,
                parent_id=parent_id,
                tool_type=tool_type,
                stride=stride,
            )
            nodes.append(node)
        return nodes

    def open_subtitles(self, subtitles_path: str, video_path: str) -> List[Dict[str, Any]]:
        """打开字幕文件"""
        subtitles_path = os.path.join(os.path.dirname(os.path.dirname(video_path)), 'subtitles', subtitles_path)
        with open(subtitles_path, 'r') as f:
            subtitles = json.load(f)
        return subtitles

    def timestamp_to_seconds(self, timestamp):
        # Split the timestamp into hours, minutes, and seconds
        h, m, s = timestamp.split(":")
        # Convert hours, minutes, and total seconds (including fractions) to float and compute total seconds
        total_seconds = int(h) * 3600 + int(m) * 60 + float(s)
        return total_seconds
    
    def insert_subtitles_into_frames(
        self,
        frames_data,
        subtitles,
        starting_timestamp_for_subtitles,
        duration
    ) -> List[Dict[str, Any]]:
        """插入字幕到帧"""
        interleaved = []
        cur_i = 0

        # 从 frames_data 中提取 frames 和 frame_timestamps
        frames = [frame["base64"] for frame in frames_data]
        frame_timestamps = [frame["timestamp"] for frame in frames_data]

        for sub in subtitles:
            # 1. 把字幕时间戳转成"相对秒数"
            if "timestamp" in sub:
                start, end = sub["timestamp"]
                if not isinstance(end, float):
                    end = duration
                start -= starting_timestamp_for_subtitles
                end -= starting_timestamp_for_subtitles
                mid = (start + end) / 2
                text = sub["text"]
            else:  # 另一种格式
                start = self.timestamp_to_seconds(sub["start"])
                end = self.timestamp_to_seconds(sub["end"])
                start -= starting_timestamp_for_subtitles
                end -= starting_timestamp_for_subtitles
                mid = (start + end) / 2
                text = sub["line"]

            # 2. 把"早于字幕中心时间"的帧先放进去
            for i, (frm, ts) in enumerate(zip(frames[cur_i:], frame_timestamps[cur_i:])):
                if ts <= mid:
                    interleaved.append({
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64," + frm}
                })
                    cur_i += 1
                else:
                    break

            # 3. 如果这段字幕时间范围内有帧覆盖，就把字幕文本插进去
            covering = any(ts < end and ts > start for ts in frame_timestamps)
            if covering:
                interleaved.append({"type": "text", "text": f"timestamp: {ts:.2f}s" + "\n" + f"subtitle: {text}" + "\n"})

        # 4. 剩余帧全部追加
        for frm, ts in zip(frames[cur_i:], frame_timestamps[cur_i:]):
            interleaved.append({"type": "image_url","image_url": {"url": "data:image/jpeg;base64," + frm}})
            interleaved.append({"type": "text", "text": f"timestamp: {ts:.2f}s" + "\n"})


        return interleaved

    

    def run(self, dataset: str, video_path: str, question: str, max_frames: int, fps: float, question_type: str = "multiple_choice",  row: pd.Series = None) -> ToTRunResult:
        """
        主要的运行方法 - BFS版本，找到答案即退出
        """
        # 重置实例状态
        self.messages = []
        self.frame_cache = {}
        
        # 重置token统计
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.api_call_count = 0
        
        # 设置视频特定的工作目录
        video_dir = os.path.dirname(video_path)
        video_filename = os.path.basename(video_path)
        video_name = os.path.splitext(video_filename)[0]
        
        # 创建线程特定的工作目录
        thread_suffix = f"_t{self.thread_id}" if self.thread_id else ""
        self.workdir = os.path.join(video_dir, f"{video_name}_work{thread_suffix}")
        ensure_dir(self.workdir)
        
        print(f"[Thread {self.thread_id}] Processing {video_name} in {self.workdir}")
        
        try:
            meta = self._video_meta(video_path)
            init_segment_frames = self._extract_frames_with_timestamps(
                    video_path, 
                    start_s=0, 
                    end_s=meta["duration"],
                    max_frames=max_frames,
                    fps=fps
                )

            frame_content = []

            if dataset == 'longvideobench':
                subtitles = self.open_subtitles(row['subtitle_path'], video_path)
                interleaved = self.insert_subtitles_into_frames(init_segment_frames, subtitles, row['starting_timestamp_for_subtitles'], row['duration'])
                frame_content = interleaved

            # 提取初始帧
            else:                 
                # 构建帧内容

                for frame_info in init_segment_frames:
                    frame_content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + frame_info["base64"]}})
                    frame_content.append({"type": "text", "text": f"timestamp: {frame_info['timestamp']}s"})

            if "multiple_choice" in question_type.lower() or "Options:" in question:
                instruction = "Make sure to put only the letter of your final chosen option inside the \\boxed{} at the end of your response."
            else:
                instruction = "Make sure to put only your final numerical answer inside the \\boxed{} at the end of your response."
            
            question += f'\n\n{instruction}'

            frame_content.append({"type": "text", "text": build_root_user_prompt(meta, question, max_paths=self.per_expand_limit)})
            
            # 初始化对话
            self.messages = [
                {"role": "system", "content": ROOT_SYSTEM_PROMPT},
                {"role": "user", "content": frame_content},
            ]

            # 根规划
            root_decision = self._chat_to_json()
            nodes: Dict[str, PathNode] = {}
            root_paths: List[str] = []

            if root_decision.get("decision") == "answer":
                all_answers = [{
                    "path_id": "root",
                    "answer": root_decision.get("direct_answer"),
                    "rationale": root_decision.get("rationale", ""),
                    "messages": self.messages
                }]
                return ToTRunResult(
                    root_paths=[], 
                    nodes=nodes, 
                    final_answer=root_decision.get("direct_answer"), 
                    terminated=True, 
                    all_answers=all_answers,
                    total_input_tokens=self.total_input_tokens,
                    total_output_tokens=self.total_output_tokens,
                    total_tokens=self.total_tokens,
                    api_call_count=self.api_call_count
                )
            
            if root_decision.get("decision") == "terminate":
                return ToTRunResult(
                    root_paths=[], 
                    nodes=nodes, 
                    final_answer=None, 
                    terminated=True, 
                    all_answers=[],
                    total_input_tokens=self.total_input_tokens,
                    total_output_tokens=self.total_output_tokens,
                    total_tokens=self.total_tokens,
                    api_call_count=self.api_call_count
                )

            proposed = root_decision.get("proposed_paths", [])[: self.per_expand_limit]
            initial_nodes = self._sanitize_paths(proposed, parent_id=None, duration=meta["duration"], depth=1)
            for n in initial_nodes:
                nodes[n.path_id] = n
                root_paths.append(n.path_id)

            # 收集所有找到的答案
            all_answers = []
            final_answer = None
            terminated = False
            confidence = None

            # 根对话历史快照
            root_history = deepcopy(self.messages)

            # BFS队列 - 使用collections.deque以提高效率
            from collections import deque
            queue = deque(root_paths)
            
            # 按深度组织节点，确保BFS顺序
            depth_nodes = {1: list(root_paths)}
            current_depth = 1
            max_reached_depth = 1

            while queue:
                # 处理当前深度的所有节点
                current_level_size = len(queue)
                next_depth_nodes = []
                
                for _ in range(current_level_size):
                    pid = queue.popleft()
                    node = nodes[pid]
                    
                    if node.depth > self.max_depth:
                        continue
             
                    # 工具调用：剪辑片段
                    try:
                        if node.tool_type in ["global", "local"]:
                            clip_res = clip_video_segment(video_path, node.start_s, node.end_s, workdir=self.workdir, tool_type=node.tool_type)
                        elif node.tool_type in ["slide"]:  
                            clip_res = clip_video_segment(video_path, node.father_start_s, node.father_end_s, workdir=self.workdir, tool_type=node.tool_type, slide=node.stride)
                        node.clip_result = clip_res
                    except Exception as e:
                        print(f"[Thread {self.thread_id}] Error clipping video segment: {e}")
                        continue

                    # 构建本地线程
                    if node.parent_id is None:
                        local_messages = deepcopy(root_history)
                    else:
                        parent = nodes[node.parent_id]
                        local_messages = deepcopy(parent.messages) if parent.messages is not None else deepcopy(root_history)

                    # 提取片段帧
                    segment_frames = self._extract_frames_with_timestamps(
                        video_path, 
                        start_s=clip_res.start_s, 
                        end_s=clip_res.end_s
                    )
                    
                    # 构建片段内容
                    segment_content = []
                    for frame_info in segment_frames:
                        segment_content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + frame_info["base64"]}})
                        if 'original_timestamp' in frame_info:
                            timestamp_text = f"timestamp: {frame_info['original_timestamp']:.2f}s"
                        else:
                            timestamp_text = f"timestamp: {frame_info.get('target_timestamp', frame_info['timestamp']):.2f}s"
                        segment_content.append({"type": "text", "text": timestamp_text})
                    
                    segment_content.append({"type": "text", "text": build_node_user_prompt(
                        path_id=node.path_id,
                        strategy=node.strategy,
                        start_s=node.start_s,
                        end_s=node.end_s,
                        clip_path=clip_res.path,
                        duration=clip_res.duration,
                        max_paths=self.per_expand_limit
                    )})

                    local_messages.append({
                        "role": "user",
                        "content": segment_content
                    })

                    # 获取节点决策
                    node_decision = self._chat_to_json(messages=local_messages)

                    # 保存本地线程到节点
                    node.messages = local_messages

                    # 更新节点状态
                    node.status = "processed"
                    node.decision = node_decision.get("decision")
                    node.rationale = node_decision.get("rationale")
                    node.confidence = node_decision.get("evidence_confidence", 0.0)

                    # 分支处理
                    if node.decision == "discard":
                        node.status = "discarded"
                        continue

                    if node.decision == "answer":
                        answer_info = {
                            "path_id": node.path_id,
                            "answer": node_decision.get("direct_answer"),
                            "rationale": node_decision.get("rationale", ""),
                            "start_s": node.start_s,
                            "end_s": node.end_s,
                            "strategy": node.strategy,
                            "depth": node.depth,
                            "evidence_confidence": node_decision.get("evidence_confidence", 0.0)
                        }
                        all_answers.append(answer_info)
                        node.direct_answer = node_decision.get("direct_answer")
                        
                        if final_answer is None:
                            final_answer = node_decision.get("direct_answer")
                            confidence = node_decision.get("evidence_confidence", 0.0)
                            
                        print(f"[Thread {self.thread_id}] Found answer from path {node.path_id} at depth {node.depth}: {node_decision.get('direct_answer')}")
                        
                        # 如果启用了early_stop，找到答案就立即返回
                        if self.early_stop:
                            print(f"[Thread {self.thread_id}] Early stopping - answer found!")
                            terminated = True
                            return ToTRunResult(
                                root_paths=root_paths, 
                                nodes=nodes, 
                                final_answer=final_answer, 
                                terminated=terminated, 
                                all_answers=all_answers, 
                                confidence=confidence,
                                total_input_tokens=self.total_input_tokens,
                                total_output_tokens=self.total_output_tokens,
                                total_tokens=self.total_tokens,
                                api_call_count=self.api_call_count
                            )
                        continue

                    if node.confidence and node.confidence < 4 and node.confidence > 0:
                        continue

                    if node.decision == "expand":
                        if node.depth >= self.max_depth:
                            print(f"[Thread {self.thread_id}] Path {node.path_id} reached max depth {self.max_depth}")
                            continue
                        
                        child_nodes = self._sanitize_paths(
                            node_decision.get("proposed_paths", [])[: self.per_expand_limit],
                            parent_id=node.path_id,
                            duration=meta["duration"],
                            depth=node.depth + 1,
                        )
                        for child in child_nodes:
                            child.father_start_s = node.start_s
                            child.father_end_s = node.end_s
                            nodes[child.path_id] = child
                            node.children.append(child.path_id)
                            next_depth_nodes.append(child.path_id)
                
                # 将下一深度的节点加入队列（保证BFS顺序）
                if next_depth_nodes:
                    queue.extend(next_depth_nodes)
                    current_depth += 1
                    if current_depth not in depth_nodes:
                        depth_nodes[current_depth] = []
                    depth_nodes[current_depth].extend(next_depth_nodes)
                    max_reached_depth = max(max_reached_depth, current_depth)
                    print(f"[Thread {self.thread_id}] Starting depth {current_depth} with {len(next_depth_nodes)} nodes")

            # 设置结果状态
            if all_answers:
                terminated = True
                print(f"[Thread {self.thread_id}] BFS exploration completed. Found {len(all_answers)} answers. Max depth reached: {max_reached_depth}")
            else:
                print(f"[Thread {self.thread_id}] BFS exploration completed. No answers found. Max depth reached: {max_reached_depth}")

            return ToTRunResult(
                root_paths=root_paths, 
                nodes=nodes, 
                final_answer=final_answer, 
                terminated=terminated, 
                all_answers=all_answers, 
                confidence=confidence,
                total_input_tokens=self.total_input_tokens,
                total_output_tokens=self.total_output_tokens,
                total_tokens=self.total_tokens,
                api_call_count=self.api_call_count
            )
            
        except Exception as e:
            print(f"[Thread {self.thread_id}] Error in run method: {e}")
            import traceback
            traceback.print_exc()
            return ToTRunResult(
                root_paths=[], 
                nodes={}, 
                final_answer=None, 
                terminated=True, 
                all_answers=[],
                total_input_tokens=self.total_input_tokens,
                total_output_tokens=self.total_output_tokens,
                total_tokens=self.total_tokens,
                api_call_count=self.api_call_count
            )

    @staticmethod
    def export_tree(result: ToTRunResult) -> Dict[str, Any]:
        """导出推理树"""
        out_nodes = {}
        for pid, n in result.nodes.items():
            out_nodes[pid] = {
                "path_id": n.path_id,
                "parent_id": n.parent_id,
                "strategy": n.strategy,
                "start_s": n.start_s,
                "end_s": n.end_s,
                "status": n.status,
                "decision": n.decision,
                "direct_answer": n.direct_answer,
                "rationale": n.rationale,
                "father_start_s": n.father_start_s,
                "father_end_s": n.father_end_s,
                "tool_type": n.tool_type,
                "stride": n.stride,
                "clip_path": n.clip_result.path if n.clip_result else None,
                "children": n.children,
                "messages": n.messages,
                "evidence_confidence": n.confidence,
            }
        
        return {
            "final_answer": result.final_answer,
            "evidence_confidence": result.confidence,
            "terminated": result.terminated,
            "root_paths": result.root_paths,
            "nodes": out_nodes,
            "all_answers": result.all_answers,
            "total_input_tokens": result.total_input_tokens,
            "total_output_tokens": result.total_output_tokens,
            "total_tokens": result.total_tokens,
            "api_call_count": result.api_call_count,
        }


