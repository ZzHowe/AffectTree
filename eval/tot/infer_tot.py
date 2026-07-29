import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tot_engine_concurrent import ConcurrentToTEngine
import pandas as pd
from threading import Lock
from tqdm import tqdm
import numpy as np
import traceback
import re
import threading

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
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
    elif hasattr(obj, 'item'):  # Handle numpy scalars
        return obj.item()
    return obj


def safe_json_dump(obj, file, **kwargs):
    """Safe JSON dump that handles numpy types"""
    converted_obj = convert_numpy_types(obj)
    json.dump(converted_obj, file, **kwargs)

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
        
        elif dataset_name == 'tomato':
            df['_index'] = df['task_type'] + '_' + df['index'].astype(str)
        
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

class ConcurrentProcessor:
    """并发处理器"""

    def __init__(self, args: argparse.Namespace, max_workers: int = 32):
        self.args = args
        self.max_workers = max_workers
        self.results_lock = Lock()
        self.progress_lock = Lock()
        self.results = []
        self.successful_count = 0
        self.failed_count = 0
        self.processed_count = 0
        self.skipped_count = 0  # 跳过的文件数量
        self.reasoning_success_count = 0  # 基于final_answer是否为None的成功计数
        self.reasoning_failure_count = 0  # 基于final_answer是否为None的失败计数
        self.config = DatasetConfig.get_config(self.args.dataset)
        self.dataset_name = self.args.dataset  # Add this line to fix the missing attribute

        self.output_dir = self._get_effective_output_dir()
        os.makedirs(self.output_dir, exist_ok=True)

        self.stats = {
            'start_time': None,
            'end_time': None,
            'total_videos': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0,  # 跳过的视频数
            'reasoning_success': 0,  # 基于final_answer的成功数
            'reasoning_failure': 0,  # 基于final_answer的失败数
            'errors': [],
            'output_directory': self.output_dir,
            'total_input_tokens': 0,  # 总input tokens
            'total_output_tokens': 0,  # 总output tokens
            'total_tokens': 0,  # 总tokens
            'avg_input_tokens': 0.0,  # 平均input tokens
            'avg_output_tokens': 0.0,  # 平均output tokens
            'avg_total_tokens': 0.0,  # 平均total tokens
        }

    def _is_numeric_question(self, row: pd.Series, question_type: str) -> bool:
        """判断是否为数字类型问题"""
        if question_type == "numeric":
            return True
        
        # 对于vsibench数据集，检查ground_truth是否为数字
        if self.dataset_name == 'vsibench':
            ground_truth = str(row.get("ground_truth", "")).strip()
            try:
                float(ground_truth)
                return True
            except:
                return False
        
        return False

    def _get_effective_output_dir(self) -> str:
        """获取有效的输出目录"""
        return os.path.abspath(Path(self.args.output_dir) if hasattr(self.args, "output_dir") and self.args.output_dir else self.args.data_dir)

    def get_video_output_dir(self, video_id: str, identifier: str = None) -> str:
        """获取特定视频的输出目录路径"""
        # 获取视频子目录前缀：优先使用命令行参数 --video_subdir_prefix，若未提供则退化为数据集名称
        # video_subdir_prefix = getattr(self.args, 'video_subdir_prefix', self.dataset_name)
        if identifier:
            return os.path.join(self.output_dir, self.dataset_name, f'{self.args.max_frames}_frames', f'{video_id}_{identifier}')
        else:
            return os.path.join(self.output_dir, self.dataset_name, f'{self.args.max_frames}_frames', f'{video_id}')

    def create_engine(self) -> ConcurrentToTEngine:
        """为每个线程创建独立的ConcurrentToTEngine实例"""
        return ConcurrentToTEngine(
            llm_model=self.args.model,
            base_url = self.args.base_url,
            api_key=self.args.api_key,
            max_depth=self.args.max_depth,
            per_expand_limit=self.args.per_expand_limit,
            temperature=self.args.temperature,
            thread_safe=True,
        )

    def check_existing_results(self, video_output_dir: str) -> bool:
        """检查是否已经存在结果文件，如果存在则跳过处理"""
        if not getattr(self.args, 'resume', False):
            return False
            
        result_file = os.path.join(video_output_dir, "result.json")
        tree_file = os.path.join(video_output_dir, "tree.json")
        
        # 检查两个文件是否都存在且不为空
        if os.path.exists(result_file) and os.path.exists(tree_file):
            try:
                # 检查文件是否可读且包含有效内容
                if os.path.getsize(result_file) > 0 and os.path.getsize(tree_file) > 0:
                    with open(result_file, 'r') as f:
                        json.load(f)  # 验证JSON格式
                    with open(tree_file, 'r') as f:
                        json.load(f)  # 验证JSON格式
                    return True
            except (json.JSONDecodeError, OSError):
                # 如果文件损坏，不跳过，重新处理
                pass
        
        return False

    def process_single_video(self, video_data: Tuple[int, pd.Series]) -> Optional[Dict[str, Any]]:
        """处理单个视频"""
        orig_idx, row = video_data
        video_path = DatasetLoader.get_video_path(row, self.args.dataset, self.config)

        try:
            video_name = None
            identifier = None
            engine = self.create_engine()
            if not os.path.exists(video_path):
                with self.progress_lock:
                    print(f"video file not found: {video_path}")
                    return None
            if self.dataset_name == 'egoschema':
                video_name = row["video_idx"] if "video_idx" in row else row.get("video", "")
           
            elif self.dataset_name == 'longvideobench':
                video_name = row["video_id"]
                identifier = row["_index"]
     
            elif self.dataset_name == 'mlvu':
                video_name = os.path.splitext(os.path.basename(row["video"]))[0]
                identifier = row["_index"]

            elif self.dataset_name == 'tomato':
                video_name = row["video"]
                identifier = row["_index"]
                
            elif self.dataset_name == 'videomme':
                video_name = row["question_id"]
                
            elif self.dataset_name == 'vsibench':
                video_name = row["scene_name"]
                identifier = row["_index"]
            


            video_output_dir = self.get_video_output_dir(video_name, identifier=identifier)
            os.makedirs(video_output_dir, exist_ok=True)

            video_idx = video_name

            if self.check_existing_results(video_output_dir):
                with self.progress_lock:
                    self.skipped_count += 1
                    thread_name = threading.current_thread().name
                    print(f"[{thread_name}] ⏭ Skipped {video_idx} (already processed) | Skipped: {self.skipped_count}")
                return None

            formatted_question, question_type = QuestionFormatter.format_question(
                row, self.dataset_name
            )

            is_numeric = self._is_numeric_question(row, question_type)
            
            # 获取真实答案
            ground_truth = GroundTruthExtractor.get_ground_truth(
                row, self.dataset_name, getattr(self.args, 'split', None)
            )

            max_frames = self.args.max_frames if hasattr(self.args, 'max_frames') else ConcurrentProcessor.CONFIGS[self.dataset_name]['max_frames']
            fps = self.args.fps if hasattr(self.args, 'fps') else ConcurrentProcessor.CONFIGS[self.dataset_name]['fps']

            result = engine.run(self.args.dataset, video_path, formatted_question, max_frames=max_frames, fps=fps, question_type=question_type, row=row)
            
            engine._cleanup_by_lru()
            # result = engine.run(video_path, formatted_question)
            tree = engine.export_tree(result)

            # 确定推理状态和成功标志
            predicted_answer = result.final_answer
            extracted_answer = AnswerExtractor.extract_answer(predicted_answer, question_type)
            # extracted_answer = predicted_answer
            reasoning_success = extracted_answer is not None
            reasoning_status = "success" if reasoning_success else "failed"

            # 创建结果条目
            entry_result = {
                "video_id": video_idx,
                "question_id": row.get("question_idx", ""),
                "video_idx": video_idx,
                "video_path": video_path,
                "question": formatted_question,
                "ground_truth_answer": ground_truth,
                "final_answer": extracted_answer,
                "reasoning_success": reasoning_success,
                "reasoning_status": reasoning_status,
                "terminated": result.terminated,
                "total_nodes": len(result.nodes),
                "all_answers": result.all_answers,
                "total_answers": len(result.all_answers),
                "reasoning_tree": tree,
                "dataset": self.dataset_name,
                "original_index": orig_idx,
                "processing_thread": threading.current_thread().name,
                "split": getattr(self.args, 'split', None),
                "total_input_tokens": result.total_input_tokens,
                "total_output_tokens": result.total_output_tokens,
                "total_tokens": result.total_tokens,
                "api_call_count": result.api_call_count
            }
            
            # 提取答案
            # extracted_answer = predicted_answer  # Use predicted_answer directly since it's already extracted
            individual_result_path = os.path.join(video_output_dir, "result.json")
            with open(individual_result_path, "w", encoding="utf-8") as f:
                safe_json_dump(entry_result, f, ensure_ascii=False, indent=2)
            
            # 保存推理树
            tree_path = os.path.join(video_output_dir, "tree.json")
            with open(tree_path, "w", encoding="utf-8") as f:
                safe_json_dump(tree, f, ensure_ascii=False, indent=2)
            
            # 更新进度信息
            with self.progress_lock:
                self.successful_count += 1
                if reasoning_success:
                    self.reasoning_success_count += 1
                else:
                    self.reasoning_failure_count += 1
                
                # 累计token统计
                self.stats['total_input_tokens'] += result.total_input_tokens
                self.stats['total_output_tokens'] += result.total_output_tokens
                self.stats['total_tokens'] += result.total_tokens
                
                thread_name = threading.current_thread().name
                print(f"[{thread_name}] ✓ Processed {video_idx} | Success: {self.successful_count}")
                print(f"[{thread_name}] Final answer: {extracted_answer} | Reasoning: {reasoning_status}")
                print(f"[{thread_name}] Ground truth: {ground_truth}")
                print(f"[{thread_name}] Nodes: {len(result.nodes)}, Answers: {len(result.all_answers)}")
                print(f"[{thread_name}] Tokens - Input: {result.total_input_tokens}, Output: {result.total_output_tokens}, Total: {result.total_tokens}")
                print(f"[{thread_name}] Reasoning Success/Failure: {self.reasoning_success_count}/{self.reasoning_failure_count}")
            
            return entry_result
            
        except Exception as e:
            error_info = {
                'video_id': row.get("video_idx", row.get("video", "unknown")),
                'original_index': orig_idx,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'thread': threading.current_thread().name
            }
            
            with self.progress_lock:
                self.failed_count += 1
                self.reasoning_failure_count += 1  # 处理异常也算推理失败
                self.stats['errors'].append(error_info)
                thread_name = threading.current_thread().name
                print(f"[{thread_name}] ✗ Error processing {row.get('video_idx', row.get('video', 'unknown'))}: {e}")
                print(f"[{thread_name}] Failed: {self.failed_count}")
                print(f"[{thread_name}] Reasoning Success/Failure: {self.reasoning_success_count}/{self.reasoning_failure_count}")
            
            return None


    def process_batch(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        self.stats['start_time'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        self.stats['start_time_seconds'] = time.time()
        self.stats['total_videos'] = len(df)

        print(f"Starting concurrent processing with {self.max_workers} workers...")
        print(f"Total videos to process: {len(df)}")

        video_data_list = [(orig_idx, row) for orig_idx, row in df.iterrows()]

        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_video = {
                executor.submit(self.process_single_video, video_data): video_data[0] for video_data in video_data_list
            }
        
            with tqdm(total=len(video_data_list), desc="Processing videos", unit="video") as pbar:
                for future in as_completed(future_to_video):
                    orig_idx = future_to_video[future]
                    try:
                        result = future.result()
                        if result is not None:
                            with self.results_lock:
                                self.results.append(result)

                        with self.progress_lock:
                            self.processed_count += 1
                            reasoning_success_rate = (self.reasoning_success_count / self.processed_count) * 100 if self.processed_count > 0 else 0
                            pbar.set_postfix({
                                'Success': self.successful_count,
                                'Failed': self.failed_count,
                                'Skipped': self.skipped_count,
                                'Rate': f"{(self.successful_count/self.processed_count)*100:.1f}%" if self.processed_count > 0 else "0%",
                                'R-Success': self.reasoning_success_count,
                                'R-Rate': f"{reasoning_success_rate:.1f}%"
                            })
                    except Exception as e:
                        with self.progress_lock:
                            self.failed_count += 1
                            self.reasoning_failure_count += 1
                            print(f"Error processing video {orig_idx}: {e}")

                    pbar.update(1)
        
        self.stats['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        self.stats['end_time_seconds'] = time.time()
        
        self.stats['successful'] = self.successful_count
        self.stats['failed'] = self.failed_count
        self.stats['skipped'] = self.skipped_count
        self.stats['reasoning_success'] = self.reasoning_success_count
        self.stats['reasoning_failure'] = self.reasoning_failure_count
        self.stats['output_directory'] = self.output_dir
        self.stats['total_videos'] = self.processed_count

        return self.results

    def save_results(self, results: List[Dict[str, Any]]):
        """保存最终结果和统计信息"""
        # 确定结果文件名
        if self.args.dataset == 'egoschema':
            split_suffix = f"_{self.args.split}" if hasattr(self.args, 'split') and self.args.split else ""
            self.dataset_name = f"egoschema"
        
        # if self.args.start_row is not None and self.args.end_row is not None:
        #     results_filename = f"{self.dataset_name}_concurrent_results_rows_{self.args.start_row}_{self.args.end_row-1}.json"
        #     stats_filename = f"{self.dataset_name}_concurrent_stats_rows_{self.args.start_row}_{self.args.end_row-1}.json"
        # else:
        results_filename = f"{self.dataset_name}_concurrent_overall_results.json"
        stats_filename = f"{self.dataset_name}_concurrent_overall_stats.json"
        
        # 保存结果
        overall_results_path = os.path.join(self.output_dir, f'{self.dataset_name}', f'{self.args.max_frames}_frames', results_filename)
        os.makedirs(os.path.dirname(overall_results_path), exist_ok=True)
        with open(overall_results_path, "w", encoding="utf-8") as f:
            safe_json_dump(results, f, ensure_ascii=False, indent=2)
        
        # 计算处理时间和速度
        total_time = float(self.stats['end_time_seconds']) - float(self.stats['start_time_seconds'])
        videos_per_second = self.stats['successful'] / total_time if total_time > 0 else 0
        
        # 计算平均token使用
        if self.stats['successful'] > 0:
            self.stats['avg_input_tokens'] = self.stats['total_input_tokens'] / self.stats['successful']
            self.stats['avg_output_tokens'] = self.stats['total_output_tokens'] / self.stats['successful']
            self.stats['avg_total_tokens'] = self.stats['total_tokens'] / self.stats['successful']
        
        # 更新统计信息
        self.stats.update({
            'total_time_seconds': total_time,
            'videos_per_second': videos_per_second,
            'success_rate': (self.stats['successful'] / self.stats['total_videos']) * 100 if self.stats['total_videos'] > 0 else 0,
            'reasoning_success_rate': (self.stats['reasoning_success'] / self.stats['total_videos']) * 100 if self.stats['total_videos'] > 0 else 0,
            'max_workers': self.max_workers,
            'split': self.args.split
        })
        
        # 保存统计信息
        stats_path = os.path.join(self.output_dir,  f'{self.dataset_name}', f'{self.args.max_frames}_frames', stats_filename)
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        with open(stats_path, "w", encoding="utf-8") as f:
            safe_json_dump(self.stats, f, ensure_ascii=False, indent=2)
        
        print(f"\n=== Concurrent Processing Summary ===")
        print(f"Total videos: {self.stats['total_videos']}")
        print(f"Successful: {self.stats['successful']}")
        print(f"Failed: {self.stats['failed']}")
        print(f"Skipped: {self.stats['skipped']}")
        print(f"Success rate: {self.stats['success_rate']:.1f}%")
        print(f"Reasoning successful: {self.stats['reasoning_success']}")
        print(f"Reasoning failed: {self.stats['reasoning_failure']}")
        print(f"Reasoning success rate: {self.stats['reasoning_success_rate']:.1f}%")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Processing speed: {videos_per_second:.2f} videos/second")
        print(f"Max workers: {self.max_workers}")
        print(f"Split: {self.args.split}")
        print(f"Resume mode: {getattr(self.args, 'resume', False)}")
        print(f"\n=== Token Usage Statistics ===")
        print(f"Total input tokens: {self.stats['total_input_tokens']:,}")
        print(f"Total output tokens: {self.stats['total_output_tokens']:,}")
        print(f"Total tokens: {self.stats['total_tokens']:,}")
        print(f"Average input tokens per video: {self.stats['avg_input_tokens']:.2f}")
        print(f"Average output tokens per video: {self.stats['avg_output_tokens']:.2f}")
        print(f"Average total tokens per video: {self.stats['avg_total_tokens']:.2f}")
        print(f"\nResults saved to: {overall_results_path}")
        print(f"Statistics saved to: {stats_path}")
        
        if self.stats['errors']:
            print(f"Errors encountered: {len(self.stats['errors'])}")
            print("Check stats file for detailed error information")

# ==================== 答案提取和验证 ====================

class AnswerExtractor:
    """答案提取器"""
    
    @staticmethod
    def extract_answer(answer_text: str, question_type: str = "multiple_choice") -> Union[str, float, None]:
        """提取答案（字母或数字）"""

        if len(answer_text) == 1:
            return answer_text
        
        # 支持提取 \boxed{}、\\boxed{}、boxed{}、oxed{} 中的内容
        patterns = [
            r'\\boxed\{([^}]*)\}',      # \boxed{}
            r'\\\\boxed\{([^}]*)\}',    # \\boxed{}
            r'boxed\{([^}]*)\}',        # boxed{}
            r'oxed\{([^}]*)\}'          # oxed{}
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, answer_text)
            if matches:
                boxed_content = matches[-1].strip()
                
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


    
def infer_tot(args: argparse.Namespace, df: pd.DataFrame):
    """推理TOT"""
    if "index" not in df:
        df = df.assign(index=range(len(df)))
    
    if "question_idx" in df:
        df["index"] = df["question_idx"]
    
    if "video_idx" in df:
        df["video"] = df["video_idx"]

    processor = ConcurrentProcessor(args, max_workers=args.max_workers)
    results = processor.process_batch(df)

    processor.save_results(results)
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default="./work", help="Base working directory")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model to use")
    parser.add_argument("--base_url", default="http://localhost:9011/v1", help="Base URL for the LLM API")
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--per_expand_limit", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--save_tree", default="./tree.json", help="Path to save the reasoning tree JSON")
    parser.add_argument("--dataset", default="egoschema", choices=["egoschema", "mlvu", "tomato", "videomme", "vsibench", "longvideobench"], help="Dataset to process")
    parser.add_argument("--max_frames", type=int, default=16)
    parser.add_argument("--fps", type=float, default=8.0)
    # specific arguments
    parser.add_argument("--data_dir", default="./egoschema_results", help="Base output directory for all results")
    parser.add_argument("--output_dir", help="Custom output directory (overrides data_dir if provided)")
    parser.add_argument("--video_subdir_prefix", default="tot", help="Prefix for individual video subdirectories")
    parser.add_argument("--split", default="subset", choices=["subset", "generation", "all"], help="EgoSchema split to process")

    # 随机采样参数
    parser.add_argument("--random", action="store_true", help="Randomly sample 700 items from Video-MME with seed 42")
    
    # 并发参数
    parser.add_argument("--max_workers", type=int, default=64, help="Maximum number of concurrent workers")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for result saving")
    
    # 续写参数
    parser.add_argument("--resume", action="store_true", help="Enable resume mode to skip already processed videos")
    
    args = parser.parse_args()

    effective_output_dir = args.output_dir if hasattr(args, "output_dir") and args.output_dir else args.data_dir
    effective_output_dir = os.path.abspath(Path(effective_output_dir))

    print(f"Starting concurrent {args.dataset} processing with {args.max_workers} workers...")
    print(f"Output directory: {effective_output_dir}")
    if args.dataset == 'egoschema':
        print(f"Split: {args.split}")
    print(f"Resume mode: {args.resume}")
    if args.random:
        print("Random sampling enabled: will sample 700 items from Video-MME with seed 42")
    
    print(f"Loading {args.dataset} dataset...")
    df = DatasetLoader.load_dataset(args.dataset, args.split)
    print(f"Loaded {len(df)} items")

    df = df.iloc[0:200]
    
    # 应用随机采样
    if args.random and args.dataset == "videomme":
        import random
        random.seed(42)
        if len(df) > 700:
            sampled_indices = random.sample(range(len(df)), 700)
            df = df.iloc[sampled_indices].reset_index(drop=True)
            print(f"Randomly sampled 700 items from Video-MME (seed=42)")
        else:
            print(f"Dataset has {len(df)} items, no sampling needed")

    results = infer_tot(args, df)
    print(f"\n=== Final Results ===")
    print(f"Total successful {args.dataset} results: {len(results)}")



if __name__ == "__main__":
    main()