# infer_without_tot.py
"""
统一的视频问答推理脚本
支持数据集: EgoSchema, MLVU, TOMATO, VideoMME, VSIBench

使用示例:
# EgoSchema
python infer_without_tot.py --dataset egoschema --split subset --k 3

# MLVU
python infer_without_tot.py --dataset mlvu --max_workers 128 --resume

# TOMATO
python infer_without_tot.py --dataset tomato --start_row 0 --end_row 100

# VideoMME
python infer_without_tot.py --dataset videomme --k 5

# VSIBench
python infer_without_tot.py --dataset vsibench --max_workers 256
"""

import os
import json
import argparse
import pandas as pd
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import List, Dict, Any, Optional, Tuple, Union
from tqdm import tqdm
import traceback
import numpy as np
import base64
import io
from PIL import Image
from moviepy.editor import VideoFileClip
import re
import string

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# ==================== 工具函数 ====================

def convert_numpy_types(obj):
    """转换numpy类型为Python原生类型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif hasattr(obj, 'item'):
        return obj.item()
    return obj


def safe_json_dump(obj, file, **kwargs):
    """安全的JSON序列化"""
    converted_obj = convert_numpy_types(obj)
    json.dump(converted_obj, file, **kwargs)


# ==================== 数据集配置 ====================

class DatasetConfig:
    """数据集配置类"""
    
    CONFIGS = {
        'egoschema': {
            'base_path': '/data/egoschema',
            'video_subdir': 'videos',
            'video_ext': '.mp4',
            'splits': {
                'subset': 'Subset/test-00000-of-00001.parquet',
                'generation': 'GENERATION/test-00000-of-00001.parquet'
            },
            'file_type': 'parquet',
            'answer_type': 'multiple_choice',
            'fps': 8.0,
            'max_frames': 32
        },
        'mlvu': {
            'base_path': '/data/video_mlvu/',
            'video_base_path': '/data/video_mlvu/MLVU',
            'data_file': 'MLVU_MCQ.tsv',
            'file_type': 'tsv',
            'answer_type': 'multiple_choice',
            'fps': 8.0,
            'max_frames': 16
        },
        'tomato': {
            'base_path': '/data/tomato',
            'video_base_path': '/data/tomato/TOMATO',
            'data_file': 'TOMATO.tsv',
            'file_type': 'tsv',
            'answer_type': 'multiple_choice',
            'fps': 8.0,
            'max_frames': 16
        },
        'videomme': {
            'base_path': '/data/Video-MME',
            'video_subdir': 'data',
            'video_ext': '.mp4',
            'data_file': 'videomme/test-00000-of-00001.parquet',
            'file_type': 'parquet',
            'answer_type': 'multiple_choice',
            'fps': 8.0,
            'max_frames': 32
        },
        'vsibench': {
            'base_path': '/data/VSI-Bench',
            'data_file': 'test-00000-of-00001.parquet',
            'file_type': 'parquet',
            'answer_type': 'mixed',  # 支持选择题和数字答案
            'fps': 8.0,
            'max_frames': 32
        },
        'longvideobench': {
            'base_path': '/data/LongVideoBench',
            'video_subdir': 'videos',
            'video_ext': '.mp4',
            'data_file': 'validation-00000-of-00001.parquet',
            'file_type': 'parquet',
            'answer_type': 'mixed',
            'fps': 8.0,
            'max_frames': 32
        }

    }
    
    @classmethod
    def get_config(cls, dataset_name: str) -> Dict[str, Any]:
        """获取数据集配置"""
        return cls.CONFIGS.get(dataset_name.lower(), {})


# ==================== 推理引擎 ====================

class UnifiedInferenceEngine:
    """统一的推理引擎"""
    
    def __init__(
        self,
        args,
        llm_model: str = "XiaomiMiMo/MiMo-VL-7B-SFT-2508",
        base_url: str = "http://localhost:9010/v1",
        temperature: float = 0.1,
        thread_safe: bool = True,
        fps: float = 8.0,
        max_frames: int = 16
    ):
    
        self.args = args
        self.llm_model = llm_model
        self.temperature = temperature
        self.thread_safe = thread_safe
        self.fps = fps
        # 使用args中的max_frames（如果有）否则使用config中的
        self.max_frames = args.max_frames if hasattr(args, 'max_frames') and args.max_frames is not None else max_frames
        self.base_url = base_url

        # OpenAI客户端
        self.client = None
        if OpenAI is not None:
            self.client = OpenAI(api_key="", base_url=self.base_url)
        
        # 帧缓存
        self.frame_cache: Dict[str, List[Dict[str, Any]]] = {}

        #最大缓存数量
        self.cache_max_size = 4
        
        # 线程锁
        if thread_safe:
            self._lock = threading.RLock()
        else:
            self._lock = None
    
    #1105实现了 cache清除，在>self.cache_max_size时自动清除最旧的项，可以调整。
    def _cleanup_by_lru(self):
        """基于 LRU 策略清理缓存"""
        while len(self.frame_cache) > self.cache_max_size:
            self.frame_cache.popitem(last=False)  # 移除最老的项

    def _extract_frames(self, video_path: str) -> List[Dict[str, Any]]:
        """提取视频帧"""
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
                frame_cache_path =  f"/data/custom_frame_cache_resize/{keyword}/{filename}_fps8.0.pkl"
                break
        if frame_cache_path !="":
            import pickle
            with open(frame_cache_path, 'rb') as file:
                frames_cache = pickle.load(file)
            self.frame_cache[video_path] = frames_cache["frames"]
            cached_frames = frames_cache["frames"]
            total_frames = min(self.max_frames, len(frames_cache["frames"]))
            sampled_frames = cached_frames
            if len(cached_frames) <= total_frames:
                sampled_frames = cached_frames
            else:
                indices = [round(i * len(cached_frames) / total_frames) for i in range(total_frames)]
                indices = [min(idx, len(cached_frames) - 1) for idx in indices]
                sampled_frames = [cached_frames[idx] for idx in indices]
        
            self.frame_cache[video_path] = sampled_frames
            # print(f"use frame cache")
            return sampled_frames
        
        frames_data = []
        
        try:
            print("use moviepy to extract frames")
            with VideoFileClip(video_path) as clip_origin:
                from moviepy.video.fx.resize import resize
                clip = resize(clip_origin, 0.75)
                duration = clip.duration
                
                total_frames = min(self.max_frames, int(duration * self.fps))
                if total_frames <= 1:
                    timestamps = [0.0]
                else:
                    timestamps = [i * duration / (total_frames - 1) for i in range(total_frames)]
                
                for timestamp in timestamps:
                    try:
                        frame = clip.get_frame(min(timestamp, duration - 0.1))
                        pil_image = Image.fromarray(frame.astype('uint8'))
                        
                        buffer = io.BytesIO()
                        pil_image.save(buffer, format='JPEG', quality=85)
                        
                        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')


                        frames_data.append({
                            "base64": img_base64,
                            "timestamp": timestamp
                        })
                    except Exception as e:
                        print(f"Warning: Failed to extract frame at {timestamp}s: {e}")
                        continue
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
        
        self.frame_cache[video_path] = frames_data
        return frames_data
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

    def open_subtitles(self, subtitles_path: str, video_path: str) -> List[Dict[str, Any]]:
        """打开字幕文件"""
        subtitles_path = os.path.join(os.path.dirname(os.path.dirname(video_path)), 'subtitles', subtitles_path)
        with open(subtitles_path, 'r') as f:
            subtitles = json.load(f)
        return subtitles

    def infer(self, video_path: str, question: str, question_type: str = "multiple_choice", row: pd.Series = None) -> Tuple[str, Dict[str, int]]:
        """执行推理，返回(答案, token统计)"""
        try:
            frames_data = self._extract_frames(video_path)
            self._cleanup_by_lru()
            if self.args.dataset == 'longvideobench':
                subtitles = self.open_subtitles(row['subtitle_path'], video_path)
                interleaved = self.insert_subtitles_into_frames(frames_data, subtitles, row['starting_timestamp_for_subtitles'], row['duration'])
            
            if not frames_data:
                return "Error: Could not extract frames from video", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            
            # 构建消息内容
            content = []
            
            # 添加帧
            for frame_info in frames_data:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64," + frame_info["base64"]}
                })
                content.append({
                    "type": "text",
                    "text": f"timestamp: {frame_info['timestamp']:.2f}s"
                })

            # print()
            
            # 根据问题类型添加不同的指令
            if "multiple_choice" in question_type.lower() or "Options:" in question:
                instruction = "Make sure to put only the letter of your final chosen option inside the \\boxed{} at the end of your response."
            else:
                instruction = "Make sure to put only your final numerical answer inside the \\boxed{} at the end of your response."
            
            if self.args.dataset == 'longvideobench':
                content = interleaved

            content.append({
                "type": "text",
                "text": f"{question}\n\n{instruction}"
            })
            

            if self.args.skip_system_prompt:
                messages = [
                    {
                        "role": "user",
                        "content": content
                    }
                ]
                
            else:
                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant that analyzes video content and answers questions accurately. Always put your final answer in \\boxed{} format."
                    },
                    {
                        "role": "user",
                        "content": content
                    }
                ]

            if self.client is None:
                return "Error: No LLM client available", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            
            response = self.client.chat.completions.create(
                model=self.llm_model,
                temperature=self.temperature,
                messages=messages,
            )
            
            # 提取token统计信息
            token_stats = {
                "input_tokens": response.usage.prompt_tokens if hasattr(response, 'usage') else 0,
                "output_tokens": response.usage.completion_tokens if hasattr(response, 'usage') else 0,
                "total_tokens": response.usage.total_tokens if hasattr(response, 'usage') else 0
            }
            
            return response.choices[0].message.content.strip(), token_stats
            
        except Exception as e:
            return f"Error: {str(e)}", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


# ==================== 答案提取和验证 ====================

class AnswerExtractor:
    """答案提取器"""
    
    @staticmethod
    def extract_answer(answer_text: str, question_type: str = "multiple_choice") -> Union[str, float, None]:
        """提取答案（字母或数字）"""

        if len(answer_text) == 1:
            return answer_text
        
        # 首先尝试提取 \boxed{} 中的内容
        boxed_pattern = r'\\boxed\{([^}]*)\}'
        boxed_matches = re.findall(boxed_pattern, answer_text)
        if boxed_matches:
            boxed_content = boxed_matches[-1].strip()
            
            # 尝试解析为数字
            try:
                numeric_match = re.search(r'-?\d+\.?\d*', boxed_content)
                if numeric_match:
                    return float(numeric_match.group())
            except:
                pass
            
            # 尝试解析为选项字母
            upper_content = boxed_content.upper()
            for char in upper_content:
                if char in 'ABCDEFGHIJK':
                    return char
            
            return boxed_content
        
        return None
    
    @staticmethod
    def calculate_mra(predicted: float, ground_truth: float) -> float:
        """
        计算 Mean Relative Accuracy (MRA)
        
        Formula: MRA = (1/10) * Σ 𝟙(|ŷ - y|/|y| < 1 - θ)
        where θ ∈ {0.5, 0.55, 0.60, ..., 0.95}
        
        Returns:
            MRA score in [0, 1]
        """
        confidence_thresholds = [0.5 + i * 0.05 for i in range(10)]
        
        # 处理真实值为0的特殊情况
        if abs(ground_truth) < 1e-10:
            accuracies = [1.0 if abs(predicted) < 0.01 else 0.0 
                         for _ in confidence_thresholds]
        else:
            # 计算相对误差率
            relative_error = abs(predicted - ground_truth) / abs(ground_truth)
            accuracies = [1.0 if relative_error < (1 - theta) else 0.0 
                         for theta in confidence_thresholds]
        
        return sum(accuracies) / len(accuracies)
    
    @staticmethod
    def check_answer(predicted: Union[str, float, None], ground_truth: str, 
                    question_type: str = "multiple_choice") -> float:
        """
        检查答案是否正确，返回分数
        
        Returns:
            - 数值题：返回 MRA score (float in [0, 1])
            - 选择题：返回 1.0 (正确) 或 0.0 (错误)
        """
        
        if predicted is None:
            return 0.0
        
        try:
            # 尝试数字比较（数值题）
            gt_numeric = float(ground_truth)
            pred_numeric = None
            
            if isinstance(predicted, (int, float)):
                pred_numeric = float(predicted)
            elif isinstance(predicted, str):
                try:
                    pred_numeric = float(predicted)
                except:
                    pass
            
            if pred_numeric is not None:
                # 返回 MRA 分数（连续值 0-1）
                mra_score = AnswerExtractor.calculate_mra(pred_numeric, gt_numeric)
                return mra_score
        
        except:
            pass
        
        # 字符串比较（选择题）
        if isinstance(predicted, str):
            is_correct = predicted.upper().strip() == ground_truth.upper().strip()
        else:
            is_correct = str(predicted).strip() == str(ground_truth).strip()
        
        return 1.0 if is_correct else 0.0


# ==================== 数据集加载器 ====================

class DatasetLoader:
    """数据集加载器"""
    
    @staticmethod
    def load_dataset(dataset_name: str, split: str = None) -> pd.DataFrame:
        """加载数据集"""
        config = DatasetConfig.get_config(dataset_name)
        
        if not config:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # 构建数据文件路径
        base_path = config['base_path']
        
        if config['file_type'] == 'parquet':
            if dataset_name == 'egoschema':
                if split not in config['splits']:
                    raise ValueError(f"Invalid split for EgoSchema: {split}")
                data_path = os.path.join(base_path, config['splits'][split])
            else:
                data_path = os.path.join(base_path, config['data_file'])
            
            df = pd.read_parquet(data_path)
            
        elif config['file_type'] == 'tsv':
            data_path = os.path.join(base_path, config['data_file'])
            df = pd.read_csv(data_path, sep='\t')
        
        else:
            raise ValueError(f"Unsupported file type: {config['file_type']}")
        
        # 添加索引列（如果不存在）
        if "index" not in df.columns:
            df = df.assign(index=range(len(df)))
        
        # 数据集特定的预处理
        df = DatasetLoader._preprocess_dataset(df, dataset_name)
        
        return df
    
    @staticmethod
    def _preprocess_dataset(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """数据集特定的预处理"""
        
        if dataset_name == 'mlvu':
            df['_index'] = df['task_type'] + '_' + df['index'].astype(str)
        
        elif dataset_name == 'vsibench':
            df['_index'] = df['dataset'] + '_' + df['id'].astype(str)
        
        elif dataset_name == 'egoschema':
            if "question_idx" in df.columns:
                df["index"] = df["question_idx"]
            if "video_idx" in df.columns:
                df["video"] = df["video_idx"]

        elif dataset_name == 'longvideobench':
            df['_index'] = df['id'].astype(str)
        
        return df
    
    @staticmethod
    def get_video_path(row: pd.Series, dataset_name: str, config: Dict) -> str:
        """获取视频路径"""
        
        if dataset_name == 'egoschema':
            video_idx = row.get("video_idx", row.get("video", ""))
            return os.path.join(config['base_path'], config['video_subdir'], 
                              video_idx + config['video_ext'])
        
        elif dataset_name == 'mlvu':
            prefix = row["prefix"].replace("./MLVU", "")
            return os.path.join(config['video_base_path'], prefix.lstrip("/"), row["video"])
        
        elif dataset_name == 'tomato':
            prefix = row["prefix"].replace(".", "")
            video_key = row["video"]
            suffix = row["suffix"]
            return config['video_base_path'] + prefix + "/" + video_key + suffix
        
        elif dataset_name == 'videomme':
            video_filename = row["videoID"] + ".mp4"
            return os.path.join(config['base_path'], config['video_subdir'], video_filename)
        
        elif dataset_name == 'vsibench':
            scene_name = row["scene_name"]
            dataset_type = row["dataset"]
            return os.path.join(config['base_path'], dataset_type, f"{scene_name}.mp4")
        
        elif dataset_name == 'longvideobench':
            video_filename = row['video_path']
            return os.path.join(config['base_path'], config['video_subdir'], video_filename)
        
        return ""


# ==================== 问题格式化器 ====================

class QuestionFormatter:
    """问题格式化器"""
    
    @staticmethod
    def format_question(row: pd.Series, dataset_name: str) -> Tuple[str, str]:
        """格式化问题，返回(formatted_question, question_type)"""
        
        question = row["question"]
        question_type = "multiple_choice"
        
        if dataset_name in ['mlvu', 'tomato', 'videomme']:
            # 处理选项
            options = QuestionFormatter._get_options(row, dataset_name)
            if options:
                question = QuestionFormatter._add_options_to_question(question, options)

        elif dataset_name in ['egoschema']:
            options = QuestionFormatter._get_options(row, dataset_name)
            if options:
                question = QuestionFormatter._add_options_to_question_without_label(question, options)

        elif dataset_name == 'longvideobench':
            options = QuestionFormatter._get_options(row, dataset_name)
            if options:
                question = QuestionFormatter._add_options_to_question(question, options)
        
        elif dataset_name == 'vsibench':
            # VSIBench可能是选择题或数字题
            options = row.get("options", None)
            ground_truth = str(row.get("ground_truth", "")).strip()
            
            # 推断问题类型
            if options is not None and QuestionFormatter._has_valid_options(options):
                question_type = "multiple_choice"
                options_list = QuestionFormatter._parse_options(options)
                question = QuestionFormatter._add_options_to_question(question, options_list)
                question += "\nAnswer with the option's letter from the given choices directly."
            elif QuestionFormatter._is_numeric_answer(ground_truth):
                question_type = "numeric"
                question += "\nAnswer with the number directly."
            else:
                question_type = "unknown"
        
        return question, question_type
    
    @staticmethod
    def _get_options(row: pd.Series, dataset_name: str) -> Optional[List[str]]:
        """获取选项列表"""
        
        if dataset_name == 'egoschema':
            if "option" in row and row["option"] is not None:
                return QuestionFormatter._parse_options(row["option"])
        
        elif dataset_name == 'mlvu':
            if "candidates" in row:
                return eval(row["candidates"])
        
        elif dataset_name == 'tomato':
            if "candidates" in row and row["candidates"] is not None:
                return QuestionFormatter._parse_options(row["candidates"])
        
        elif dataset_name == 'videomme':
            options = row.get("options")
            if options is not None and len(options) > 0:
                return list(options)

        elif dataset_name == 'longvideobench':
            option0 = row.get("option0")
            option1 = row.get("option1")
            option2 = row.get("option2")
            option3 = row.get("option3")
            option4 = row.get("option4")
            return [option0, option1, option2, option3, option4]
        
        return None
    
    @staticmethod
    def _parse_options(options: Any) -> List[str]:
        """解析选项为列表"""
        if isinstance(options, str):
            try:
                return eval(options)
            except:
                return [options]
        elif isinstance(options, np.ndarray):
            return options.tolist()
        elif isinstance(options, (list, tuple)):
            return list(options)
        else:
            return [options] if options else []
    
    @staticmethod
    def _has_valid_options(options: Any) -> bool:
        """检查是否有有效选项"""
        if options is None:
            return False
        if isinstance(options, (list, tuple)):
            return len(options) > 0
        if isinstance(options, np.ndarray):
            return options.size > 0
        if isinstance(options, str):
            return bool(options.strip())
        return False
    
    @staticmethod
    def _is_numeric_answer(answer: str) -> bool:
        """检查答案是否为数字"""
        try:
            float(answer)
            return True
        except:
            return False
    
    @staticmethod
    def _add_options_to_question(question: str, options: List[str]) -> str:
        """将选项添加到问题中"""
        if not options:
            return question
        
        option_lines = []
        for i, option in enumerate(options):
            letter = chr(ord("A") + i)
            option_lines.append(f"{letter}: {option}")
        
        return question + "\nOptions:\n" + "\n".join(option_lines)
    
    @staticmethod
    def _add_options_to_question_without_label(question: str, options: List[str]) -> str:
        """将选项添加到问题中"""
        if not options:
            return question
        
        option_lines = []
        for i, option in enumerate(options):
            option_lines.append(f"{option}")
        
        return question + "\nOptions:\n" + "\n".join(option_lines)


# ==================== 真值提取器 ====================

class GroundTruthExtractor:
    """真值提取器"""
    
    @staticmethod
    def get_ground_truth(row: pd.Series, dataset_name: str, split: str = None) -> str:
        """获取真实答案"""
        
        if dataset_name == 'egoschema':
            if split == "subset":
                if "answer" in row:
                    answer_idx = int(row["answer"])
                    return chr(ord("A") + answer_idx)
            else:
                return "F"  # generation模式
        
        elif dataset_name == 'mlvu':
            answer = row["answer"]
            candidates = eval(row["candidates"])
            # 找到答案对应的字母
            for idx, option in enumerate(candidates):
                if option == answer:
                    return chr(ord("A") + idx)
            return ""
        
        elif dataset_name == 'tomato':
            answer_idx = row.get("answer", 0)
            return chr(ord("A") + answer_idx)
        
        elif dataset_name == 'videomme':
            return row.get("answer", "").strip().upper()
        
        elif dataset_name == 'vsibench':
            return str(row.get("ground_truth", "")).strip()

        elif dataset_name == 'longvideobench':
            answer_idx = row.get("correct_choice", 0)
            return chr(ord("A") + answer_idx)
        
        return ""


# ==================== 并发处理器 ====================

class ConcurrentProcessor:
    """统一的并发处理器"""
    
    def __init__(self, args, max_workers: int = 32):
        self.args = args
        self.dataset_name = args.dataset
        self.max_workers = max_workers
        
        # 锁
        self.results_lock = Lock()
        self.progress_lock = Lock()
        
        # 统计
        self.results = []
        self.successful_count = 0
        self.failed_count = 0
        
        # 新增：分数统计
        self.all_scores = []  # 所有样本的分数（0-1之间）
        self.numeric_scores = []  # 数值题的 MRA 分数
        self.choice_scores = []   # 选择题的分数（0或1）
        
        # 新增：token统计
        self.input_tokens = []  # 所有样本的输入tokens
        self.output_tokens = []  # 所有样本的输出tokens
        self.total_tokens = []  # 所有样本的总tokens
        
        # 配置
        self.config = DatasetConfig.get_config(self.dataset_name)
        
        # 输出目录 - 已修改，包含max_frames
        max_frames_str = str(args.max_frames) if hasattr(args, 'max_frames') and args.max_frames is not None else str(self.config.get('max_frames', 16))
        self.output_dir = os.path.join(os.path.abspath(args.output_dir), f"{self.dataset_name}_{max_frames_str}frame")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 统计信息
        self.stats = {
            'start_time': None,
            'end_time': None,
            'total_items': 0,
            'successful': 0,
            'failed': 0,
            'dataset': self.dataset_name,
            'output_directory': self.output_dir
        }
    
    def create_engine(self) -> UnifiedInferenceEngine:
        """创建推理引擎"""
        return UnifiedInferenceEngine(
            args=self.args,
            llm_model=self.args.model,
            base_url=self.args.base_url,
            temperature=self.args.temperature,
            thread_safe=True,
            fps=self.config.get('fps', 8.0),
            max_frames=self.args.max_frames if self.args.max_frames is not None else self.config.get('max_frames', 16)
        )
    
    def should_skip(self, identifier: str) -> bool:
        """检查是否应该跳过（已处理）"""
        if not getattr(self.args, 'resume', False):
            return False
        
        output_dir = self.get_item_output_dir(identifier)
        result_path = os.path.join(output_dir, "result.json")
        
        if os.path.exists(result_path):
            try:
                with open(result_path, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)
                if isinstance(result_data, dict) and 'predicted_answer' in result_data:
                    return True
            except:
                pass
        
        return False
    
    def get_item_output_dir(self, identifier: str) -> str:
        """获取单个项目的输出目录"""
        # 获取实际的max_frames值
        max_frames_value = self.args.max_frames if self.args.max_frames is not None else self.config.get('max_frames', 16)
        return os.path.join(self.output_dir, f"{self.dataset_name}_{identifier}_{max_frames_value}frame")
    
    def load_existing_result(self, identifier: str) -> Optional[Dict[str, Any]]:
        """加载已存在的结果"""
        try:
            output_dir = self.get_item_output_dir(identifier)
            result_path = os.path.join(output_dir, "result.json")
            with open(result_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return None
    
    def _is_numeric_question(self, row: pd.Series, question_type: str) -> bool:
        """判断是否为数值问题"""
        if question_type == "numeric":
            return True
        
        # VSIBench 特定任务判断
        if self.dataset_name == 'vsibench':
            task_type = row.get('task_type', '')
            numeric_tasks = ['Object Size', 'Room Size', 'Absolute Distance']
            return task_type in numeric_tasks
        
        # 尝试将 ground_truth 转换为数字
        try:
            float(str(row.get('ground_truth', '')))
            # 如果没有选项，也认为是数值题
            options = row.get('options', None)
            if options is None or not QuestionFormatter._has_valid_options(options):
                return True
        except:
            pass
        
        return False
    
    def process_single_item(self, item_data: Tuple[int, pd.Series]) -> Optional[Dict[str, Any]]:
        """处理单个项目"""
        orig_idx, row = item_data
        
        try:
            # 获取标识符
            identifier = self.get_identifier(row)
            
            # 检查是否跳过
            if self.should_skip(identifier):
                existing_result = self.load_existing_result(identifier)
                if existing_result:
                    with self.progress_lock:
                        self.successful_count += 1
                        
                        # 恢复分数统计
                        score = existing_result.get('score', 0.0)
                        is_numeric = existing_result.get('is_numeric', False)
                        
                        self.all_scores.append(score)
                        if is_numeric:
                            self.numeric_scores.append(score)
                        else:
                            self.choice_scores.append(score)
                        
                        # 恢复token统计
                        self.input_tokens.append(existing_result.get('input_tokens', 0))
                        self.output_tokens.append(existing_result.get('output_tokens', 0))
                        self.total_tokens.append(existing_result.get('total_tokens', 0))
                        
                        print(f"⏭ Skipped {identifier} (already processed)")
                    return existing_result
            
            # 创建推理引擎
            engine = self.create_engine()
            
            # 获取视频路径
            video_path = DatasetLoader.get_video_path(row, self.dataset_name, self.config)
            
            if not os.path.exists(video_path):
                with self.progress_lock:
                    print(f"Video not found: {video_path}")
                return None
            
            # 格式化问题
            formatted_question, question_type = QuestionFormatter.format_question(
                row, self.dataset_name
            )
            
            # 判断是否为数值题
            is_numeric = self._is_numeric_question(row, question_type)
            
            # 获取真实答案
            ground_truth = GroundTruthExtractor.get_ground_truth(
                row, self.dataset_name, getattr(self.args, 'split', None)
            )
            # 推理（返回答案和token统计）
            predicted_answer, token_stats = engine.infer(video_path, formatted_question, question_type, row)
            
            # 提取答案
            extracted_answer = AnswerExtractor.extract_answer(predicted_answer, question_type)
            
            # 计算分数（选择题返回0/1，数值题返回MRA）
            score = AnswerExtractor.check_answer(extracted_answer, ground_truth, question_type)
            
            # 二元判断（用于显示）
            is_correct = score > 0.5 if is_numeric else score == 1.0
            
            # 创建结果
            result = {
                "identifier": identifier,
                "video_path": video_path,
                "question": row["question"],
                "formatted_question": formatted_question,
                "question_type": question_type,
                "is_numeric": is_numeric,
                "ground_truth": ground_truth,
                "predicted_answer": predicted_answer,
                "extracted_answer": extracted_answer,
                "score": float(score),  # 统一的分数字段
                "is_correct": is_correct,  # 二元判断
                "original_index": orig_idx,
                "dataset": self.dataset_name,
                # Token统计
                "input_tokens": token_stats.get("input_tokens", 0),
                "output_tokens": token_stats.get("output_tokens", 0),
                "total_tokens": token_stats.get("total_tokens", 0)
            }
            
            # 添加数据集特定字段
            result.update(self.get_dataset_specific_fields(row))
            
            # 保存单个结果
            if getattr(self.args, 'save_individual', True):
                output_dir = self.get_item_output_dir(identifier)
                os.makedirs(output_dir, exist_ok=True)
                result_path = os.path.join(output_dir, "result.json")
                with open(result_path, "w", encoding="utf-8") as f:
                    safe_json_dump(result, f, ensure_ascii=False, indent=2)
            
            # 更新统计
            with self.progress_lock:
                self.successful_count += 1
                self.all_scores.append(score)
                
                if is_numeric:
                    self.numeric_scores.append(score)
                else:
                    self.choice_scores.append(score)
                
                # 更新token统计
                self.input_tokens.append(token_stats.get("input_tokens", 0))
                self.output_tokens.append(token_stats.get("output_tokens", 0))
                self.total_tokens.append(token_stats.get("total_tokens", 0))
                
                # 计算平均分
                mean_score = sum(self.all_scores) / len(self.all_scores) if self.all_scores else 0
                mean_numeric = sum(self.numeric_scores) / len(self.numeric_scores) if self.numeric_scores else 0
                mean_choice = sum(self.choice_scores) / len(self.choice_scores) if self.choice_scores else 0
                
                # 计算平均token
                mean_input_tokens = sum(self.input_tokens) / len(self.input_tokens) if self.input_tokens else 0
                mean_output_tokens = sum(self.output_tokens) / len(self.output_tokens) if self.output_tokens else 0
                
                # 打印信息
                print(f"✓ {identifier}")
                if is_numeric:
                    print(f"  [Numeric] Pred: {extracted_answer} | GT: {ground_truth} | MRA: {score:.3f}")
                else:
                    print(f"  [Choice] Pred: {extracted_answer} | GT: {ground_truth} | Correct: {is_correct}")
                print(f"  Overall Mean: {mean_score:.3f} | Numeric: {mean_numeric:.3f} ({len(self.numeric_scores)}) | Choice: {mean_choice:.3f} ({len(self.choice_scores)})")
                print(f"  Tokens - Input: {mean_input_tokens:.1f} | Output: {mean_output_tokens:.1f}")
            
            return result
            
        except Exception as e:
            with self.progress_lock:
                self.failed_count += 1
                print(f"✗ Error processing item {orig_idx}: {e}")
                print(traceback.format_exc())
            return None
    
    def get_identifier(self, row: pd.Series) -> str:
        """获取项目标识符"""
        if self.dataset_name == 'egoschema':
            return row.get("video_idx", row.get("video", str(row.get("index", ""))))
        elif self.dataset_name == 'mlvu':
            return row.get("_index", str(row.get("index", "")))
        elif self.dataset_name == 'tomato':
            return row.get("video", str(row.get("index", "")))
        elif self.dataset_name == 'videomme':
            return row.get("question_id", row.get("videoID", str(row.get("index", ""))))
        elif self.dataset_name == 'vsibench':
            return row.get("_index", row.get("scene_name", str(row.get("index", ""))))
        elif self.dataset_name == 'longvideobench':
            return row.get("_index", str(row.get("id", "")))
        return str(row.get("index", ""))
    
    def get_dataset_specific_fields(self, row: pd.Series) -> Dict[str, Any]:
        """获取数据集特定字段"""
        fields = {}
        
        if self.dataset_name == 'mlvu':
            fields['task_type'] = row.get('task_type', '')
            fields['video_name'] = os.path.splitext(os.path.basename(row["video"]))[0]
        
        elif self.dataset_name == 'tomato':
            fields['task_type'] = row.get('task_type', '')
            fields['demonstration_type'] = row.get('demonstration_type', '')
        
        elif self.dataset_name == 'videomme':
            fields['domain'] = row.get('domain', '')
            fields['sub_category'] = row.get('sub_category', '')
            fields['task_type'] = row.get('task_type', '')
            fields['duration'] = row.get('duration', '')
        
        elif self.dataset_name == 'vsibench':
            fields['scene_name'] = row.get('scene_name', '')
            fields['dataset_type'] = row.get('dataset', '')
            fields['task_type'] = row.get('task_type', '')
        
        return fields
    
    def process_batch(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """批量处理"""
        self.stats['start_time'] = time.time()
        self.stats['total_items'] = len(df)
        
        print(f"\n{'='*60}")
        print(f"Starting {self.dataset_name.upper()} inference")
        print(f"Total items: {len(df)}")
        print(f"Max workers: {self.max_workers}")
        if getattr(self.args, 'resume', False):
            print("Resume mode: ENABLED")
        print(f"{'='*60}\n")
        
        # 准备数据
        item_data_list = [(orig_idx, row) for orig_idx, row in df.iterrows()]
        
        # 并发处理
        with ThreadPoolExecutor(max_workers=self.max_workers, 
                              thread_name_prefix=f"{self.dataset_name}Worker") as executor:
            future_to_item = {
                executor.submit(self.process_single_item, item_data): item_data[0]
                for item_data in item_data_list
            }
            
            with tqdm(total=len(item_data_list), desc="Processing", unit="item") as pbar:
                for future in as_completed(future_to_item):
                    try:
                        result = future.result()
                        if result is not None:
                            with self.results_lock:
                                self.results.append(result)
                        
                        with self.progress_lock:
                            mean_score = sum(self.all_scores) / len(self.all_scores) if self.all_scores else 0
                            
                            pbar.set_postfix({
                                'Success': self.successful_count,
                                'Failed': self.failed_count,
                                'Score': f"{mean_score:.3f}",
                                'N': len(self.numeric_scores),
                                'C': len(self.choice_scores)
                            })
                    except Exception as e:
                        with self.progress_lock:
                            self.failed_count += 1
                            print(f"Future error: {e}")
                    
                    pbar.update(1)
        
        self.stats['end_time'] = time.time()
        self.stats['successful'] = self.successful_count
        self.stats['failed'] = self.failed_count
        
        return self.results
    
    def _calculate_task_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        计算任务级别统计（仅用于VSI-Bench的参考信息）
        
        Returns:
            {
                'Object Count': {'mean': 0.85, 'count': 100},
                'Absolute Distance': {'mean': 0.45, 'count': 100},
                ...
            }
        """
        from collections import defaultdict
        
        task_groups = defaultdict(list)
        
        for result in results:
            task_type = result.get('task_type', 'Unknown')
            score = result.get('score', 0.0)
            task_groups[task_type].append(score)
        
        # 计算每个任务的统计
        task_stats = {}
        for task_type, scores in task_groups.items():
            task_stats[task_type] = {
                'mean': sum(scores) / len(scores) if scores else 0.0,
                'count': len(scores)
            }
        
        return task_stats
    
    def save_results(self, results: List[Dict[str, Any]]):
        """保存结果"""
        # 获取实际的max_frames值
        max_frames_value = self.args.max_frames if self.args.max_frames is not None else self.config.get('max_frames', 16)

        # 文件名
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_filename = f"{self.dataset_name}_results_{timestamp}.json"
        stats_filename = f"{self.dataset_name}_stats_{timestamp}.json"
        
        # 保存结果
        results_path = os.path.join(self.output_dir, results_filename)
        with open(results_path, "w", encoding="utf-8") as f:
            safe_json_dump(results, f, ensure_ascii=False, indent=2)
        
        # 计算统计
        total_time = self.stats['end_time'] - self.stats['start_time']
        items_per_second = self.stats['successful'] / total_time if total_time > 0 else 0
        
        # 计算分数（直接对所有样本求平均）
        mean_score = sum(self.all_scores) / len(self.all_scores) if self.all_scores else 0.0
        mean_numeric = sum(self.numeric_scores) / len(self.numeric_scores) if self.numeric_scores else 0.0
        mean_choice = sum(self.choice_scores) / len(self.choice_scores) if self.choice_scores else 0.0
        
        # 计算二元准确率（用于对比）
        binary_correct = sum(1 for s in self.all_scores if s > 0.5)
        binary_accuracy = (binary_correct / len(self.all_scores)) * 100 if self.all_scores else 0
        
        # 计算token平均值
        mean_input_tokens = sum(self.input_tokens) / len(self.input_tokens) if self.input_tokens else 0
        mean_output_tokens = sum(self.output_tokens) / len(self.output_tokens) if self.output_tokens else 0
        mean_total_tokens = sum(self.total_tokens) / len(self.total_tokens) if self.total_tokens else 0
        
        self.stats.update({
            'total_time_seconds': total_time,
            'items_per_second': items_per_second,
            'max_workers': self.max_workers,
            'model': self.args.model,
            
            # 分数统计（样本级别，不做任务平均）
            'mean_score': mean_score,
            'numeric_count': len(self.numeric_scores),
            'choice_count': len(self.choice_scores),
            'mean_numeric_score': mean_numeric,
            'mean_choice_score': mean_choice,
            
            # 二元准确率（参考）
            'binary_accuracy': binary_accuracy,
            
            # Token统计
            'mean_input_tokens': mean_input_tokens,
            'mean_output_tokens': mean_output_tokens,
            'mean_total_tokens': mean_total_tokens,
            'total_input_tokens': sum(self.input_tokens),
            'total_output_tokens': sum(self.output_tokens),
            'total_tokens_sum': sum(self.total_tokens),
            
            # 所有分数
            'all_scores': self.all_scores,
            'numeric_scores': self.numeric_scores,
            'choice_scores': self.choice_scores
        })
        
        # VSI-Bench：额外计算任务级别统计（仅用于参考，不作为主要指标）
        if self.dataset_name == 'vsibench':
            task_stats = self._calculate_task_stats(results)
            self.stats['task_level_stats'] = task_stats
        
        # 保存统计
        stats_path = os.path.join(self.output_dir, stats_filename)
        with open(stats_path, "w", encoding="utf-8") as f:
            safe_json_dump(self.stats, f, ensure_ascii=False, indent=2)
        
        # 打印总结
        print(f"\n{'='*60}")
        print(f"{self.dataset_name.upper()} Inference Summary")
        print(f"{'='*60}")
        print(f"Model: {self.args.model}")
        print(f"Total items: {self.stats['total_items']}")
        print(f"Successful: {self.stats['successful']}")
        print(f"Failed: {self.stats['failed']}")
        
        # 主要指标：样本级别平均分
        print(f"\n--- Main Score (Sample-Level Average) ---")
        print(f"Overall Mean Score: {mean_score*100:.2f}%")
        print(f"  Numeric ({len(self.numeric_scores)} items): {mean_numeric*100:.2f}% (MRA)")
        print(f"  Choice ({len(self.choice_scores)} items): {mean_choice*100:.2f}% (Accuracy)")
        print(f"Binary Accuracy (score>0.5): {binary_accuracy:.2f}%")
        
        # Token统计
        print(f"\n--- Token Statistics ---")
        print(f"Average Input Tokens: {mean_input_tokens:.1f}")
        print(f"Average Output Tokens: {mean_output_tokens:.1f}")
        print(f"Average Total Tokens: {mean_total_tokens:.1f}")
        print(f"Total Input Tokens: {sum(self.input_tokens)}")
        print(f"Total Output Tokens: {sum(self.output_tokens)}")
        print(f"Total Tokens: {sum(self.total_tokens)}")
        
        # VSI-Bench：任务级别统计（仅供参考）
        if self.dataset_name == 'vsibench':
            print(f"\n--- Task-Level Statistics (For Reference) ---")
            task_stats = self.stats.get('task_level_stats', {})
            
            task_order = [
                'Object Count',
                'Absolute Distance',
                'Object Size', 
                'Room Size',
                'Relative Distance',
                'Relative Direction',
                'Route Plan',
                'Appearance Order'
            ]
            
            for task in task_order:
                if task in task_stats:
                    info = task_stats[task]
                    is_mra = task in ['Absolute Distance', 'Object Size', 'Room Size']
                    metric = 'MRA' if is_mra else 'Acc'
                    print(f"  {task:20s}: {info['mean']*100:5.1f}% ({metric}, n={info['count']})")
        
        print(f"\nTotal time: {total_time:.2f}s")
        print(f"Speed: {items_per_second:.2f} items/s")
        print(f"\nResults: {results_path}")
        print(f"Stats: {stats_path}")
        print(f"{'='*60}\n")


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(
        description="Unified Video Question Answering Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # EgoSchema
  python infer_without_tot.py --dataset egoschema --split subset --model Qwen/Qwen2.5-VL-32B-Instruct --output_dir /TreeR/eval/inference_results/egoschema
  
  # MLVU with resume
  python infer_without_tot.py --dataset mlvu --resume --max_workers 128
  
  # TOMATO
  python infer_without_tot.py --dataset tomato --start_row 0 --end_row 100
  
  # VideoMME
  python infer_without_tot.py --dataset videomme --k 5
  
  # VSIBench
  python infer_without_tot.py --dataset vsibench --max_workers 256
        """
    )
    
    # 数据集参数
    parser.add_argument("--dataset", required=True,
                       choices=['egoschema', 'mlvu', 'tomato', 'videomme', 'vsibench', 'longvideobench'],
                       help="Dataset name")
    parser.add_argument("--split", default="subset",
                       choices=['subset', 'generation'],
                       help="Split for EgoSchema (ignored for other datasets)")
    
    # 模型参数
    parser.add_argument("--model", default="XiaomiMiMo/MiMo-VL-7B-SFT-2508",
                       help="Model name")
    parser.add_argument("--base_url", default="http://localhost:9010/v1",
                       help="API base URL")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="Temperature")
    
    # 数据范围
    parser.add_argument("--start_row", type=int,
                       help="Start row (inclusive)")
    parser.add_argument("--end_row", type=int,
                       help="End row (exclusive)")
    
    # 输出参数
    parser.add_argument("--output_dir", default="./inference_results",
                       help="Output directory")
    parser.add_argument("--save_individual", action="store_true", default=True,
                       help="Save individual results")
    parser.add_argument("--resume", action="store_true",
                       help="Resume mode: skip processed items")
    
    # 并发参数
    parser.add_argument("--max_workers", type=int, default=48,
                       help="Max workers")

    parser.add_argument("--max_frames", type=int, default=None,
                       help="Maximum number of frames to extract from video (overrides dataset config)")

    parser.add_argument("--skip_system_prompt", action="store_true", default=False,
                       help="Skip system prompt")
    
    # Pass@K (未来扩展)
    parser.add_argument("--k", type=int, default=1,
                       help="K value for Pass@K (currently only k=1 supported)")
    args = parser.parse_args()
    
    # 加载数据集
    print(f"Loading {args.dataset} dataset...")
    df = DatasetLoader.load_dataset(args.dataset, args.split)
    print(f"Loaded {len(df)} items")

    
    # 应用行数范围
    if args.start_row is not None and args.end_row is not None:
        print(f"Filtering rows {args.start_row}-{args.end_row-1}")
        df = df.iloc[args.start_row:args.end_row]
        print(f"Filtered to {len(df)} items")
    
    # 确定worker数量
    max_workers = min(args.max_workers, len(df))
    
    # 创建处理器
    processor = ConcurrentProcessor(args, max_workers=max_workers)
    
    # 处理
    results = processor.process_batch(df)
    
    # 保存结果
    processor.save_results(results)
    
    print("\n✓ All done!")


if __name__ == "__main__":
    main()
