import json
import os
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import threading
from queue import Queue
import time
import multiprocessing as mp

from moviepy import *
from PIL import Image
import io
import base64

def extract_initial_frames(video_path: str, fps: float = 10.0) -> List[Dict[str, Any]]:
    """
    Extract frames from video at specified fps and store them with timestamps.
    
    Args:
        video_path: Path to the video file
        fps: Frames per second to extract (default: 10.0)
        
    Returns:
        List of dictionaries containing base64 encoded frames and timestamps
    """
    frames_data = []
    
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
        
    return frames_data


def get_frame_cache_path_worker(args: Tuple[str, str, str, float]) -> Tuple[str, str, bool]:
    """
    Worker function for multiprocessing frame cache generation.
    
    Args:
        args: Tuple of (video_path, video_id, cache_dir, fps)
        
    Returns:
        Tuple of (video_path, cache_path, success_flag)
    """
    video_path, video_id, cache_dir, fps = args
    
    try:
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Generate a unique filename based on video path and fps
        cache_filename = f"{video_id}_fps{fps}.json"
        cache_path = os.path.join(cache_dir, cache_filename)
        
        # Check if cache already exists
        if os.path.exists(cache_path):
            return video_path, cache_path, True
        
        # Generate frame cache
        print(f"[PID {os.getpid()}] Extracting frames for {video_path}...")
        frames_data = extract_initial_frames(video_path, fps=fps)
        
        # Save to JSON file
        cache_data = {
            "video_path": video_path,
            "fps": fps,
            "total_frames": len(frames_data),
            "frames": frames_data
        }
        
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f)
        
        print(f"[PID {os.getpid()}] Frame cache saved: {cache_path} ({len(frames_data)} frames)")
        
        return video_path, cache_path, True
        
    except Exception as e:
        print(f"[PID {os.getpid()}] Error processing {video_path}: {e}")
        return video_path, "", False


def process_jsonl_with_frame_cache_multithread(
    input_jsonl_path: str, 
    output_jsonl_path: str, 
    cache_dir: str = "./frame_cache",
    fps: float = 10.0,
    video_path_key: str = "video_path",
    frame_cache_key: str = "frame_cache_path",
    max_workers: int = None,
    use_processes: bool = True
) -> Dict[str, str]:
    """
    Process a JSONL file by adding frame cache paths to each entry using multiprocessing/multithreading.
    
    Args:
        input_jsonl_path: Path to input JSONL file
        output_jsonl_path: Path to output JSONL file
        cache_dir: Directory to save frame cache files
        fps: Frames per second for frame extraction
        video_path_key: Key name for video path in the dict
        frame_cache_key: Key name for frame cache path to add
        max_workers: Maximum number of workers (default: CPU count)
        use_processes: If True, use ProcessPoolExecutor; if False, use ThreadPoolExecutor
        
    Returns:
        Dictionary mapping video paths to frame cache paths
    """
    
    if max_workers is None:
        max_workers = mp.cpu_count()
    
    print(f"Using {'ProcessPoolExecutor' if use_processes else 'ThreadPoolExecutor'} with {max_workers} workers")
    
    # First pass: collect all unique videos
    unique_videos = {}  # video_path -> video_id
    all_data = []
    
    print("Loading JSONL file...")
    with open(input_jsonl_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                all_data.append(data)
                
                if video_path_key in data:
                    video_path = data[video_path_key]
                    video_id = data.get("videoID", f"video_{line_num}")
                    unique_videos[video_path] = video_id
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                all_data.append({"_original_line": line.strip(), "_parse_error": True})
    
    print(f"Found {len(unique_videos)} unique videos in {len(all_data)} entries")
    
    # Prepare tasks for multiprocessing
    tasks = [(video_path, video_id, cache_dir, fps) 
             for video_path, video_id in unique_videos.items()]
    
    # Process videos in parallel
    video_to_cache_map = {}
    
    if use_processes:
        executor_class = ProcessPoolExecutor
    else:
        executor_class = ThreadPoolExecutor
    
    with executor_class(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_video = {
            executor.submit(get_frame_cache_path_worker, task): task[0] 
            for task in tasks
        }
        
        # Process completed tasks
        for future in tqdm(as_completed(future_to_video), total=len(tasks), desc="Processing videos"):
            video_path = future_to_video[future]
            try:
                video_path_result, cache_path, success = future.result()
                if success:
                    video_to_cache_map[video_path_result] = cache_path
                else:
                    print(f"Failed to process: {video_path_result}")
            except Exception as e:
                print(f"Error with video {video_path}: {e}")
    
    # Second pass: write output with cache paths
    print("Writing output file...")
    with open(output_jsonl_path, 'w') as output_file:
        for data in tqdm(all_data, desc="Writing output"):
            try:
                if "_parse_error" in data:
                    # Write original line if there was a parse error
                    output_file.write(data["_original_line"] + '\n')
                    continue
                
                if video_path_key in data:
                    video_path = data[video_path_key]
                    if video_path in video_to_cache_map:
                        data[frame_cache_key] = video_to_cache_map[video_path]
                
                output_file.write(json.dumps(data) + '\n')
                
            except Exception as e:
                print(f"Error writing data: {e}")
                continue
    
    print(f"\nProcessing complete!")
    print(f"Processed {len(video_to_cache_map)} unique videos")
    print(f"Output saved to: {output_jsonl_path}")
    
    return video_to_cache_map


def split_jsonl_file(input_file: str, output_dir: str, num_splits: int) -> List[str]:
    """
    Split a JSONL file into multiple smaller files for parallel processing.
    
    Args:
        input_file: Path to input JSONL file
        output_dir: Directory to save split files
        num_splits: Number of files to split into
        
    Returns:
        List of output file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Count total lines
    with open(input_file, 'r') as f:
        total_lines = sum(1 for _ in f)
    
    lines_per_file = (total_lines + num_splits - 1) // num_splits
    
    output_files = []
    
    with open(input_file, 'r') as f:
        for i in range(num_splits):
            output_file = os.path.join(output_dir, f"split_{i:03d}.jsonl")
            output_files.append(output_file)
            
            with open(output_file, 'w') as out_f:
                for _ in range(lines_per_file):
                    line = f.readline()
                    if not line:
                        break
                    out_f.write(line)
    
    print(f"Split {input_file} into {len(output_files)} files")
    return output_files


def merge_jsonl_files(input_files: List[str], output_file: str):
    """
    Merge multiple JSONL files into one.
    
    Args:
        input_files: List of input JSONL file paths
        output_file: Path to merged output file
    """
    with open(output_file, 'w') as out_f:
        for input_file in input_files:
            if os.path.exists(input_file):
                with open(input_file, 'r') as in_f:
                    for line in in_f:
                        out_f.write(line)
    
    print(f"Merged {len(input_files)} files into {output_file}")


def validate_processed_jsonl(jsonl_path: str, video_path_key: str = "video_path", frame_cache_key: str = "frame_cache_path"):
    """
    Validate that all entries in processed JSONL have frame cache paths and files exist.
    """
    missing_cache_files = []
    missing_keys = []
    total_entries = 0
    
    with open(jsonl_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                total_entries += 1
                
                if frame_cache_key not in data:
                    missing_keys.append(line_num)
                    continue
                
                cache_path = data[frame_cache_key]
                if not os.path.exists(cache_path):
                    missing_cache_files.append((line_num, cache_path))
                    
            except json.JSONDecodeError:
                continue
    
    print(f"Validation Results:")
    print(f"Total entries: {total_entries}")
    print(f"Missing '{frame_cache_key}' key: {len(missing_keys)} entries")
    print(f"Missing cache files: {len(missing_cache_files)} entries")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process videos with multiprocessing')
    parser.add_argument('--input', required=True, help='Input JSONL file path')
    parser.add_argument('--output', required=True, help='Output JSONL file path')
    parser.add_argument('--cache-dir', default='./frame_cache', help='Cache directory')
    parser.add_argument('--fps', type=float, default=10.0, help='Frames per second')
    parser.add_argument('--workers', type=int, default=None, help='Number of workers')
    parser.add_argument('--use-threads', action='store_true', help='Use threads instead of processes')
    parser.add_argument('--split-mode', action='store_true', help='Use file splitting mode')
    parser.add_argument('--num-splits', type=int, default=4, help='Number of file splits')
    
    args = parser.parse_args()
    
    if args.split_mode:
        # Split file mode
        print("Using split file mode...")
        split_dir = f"{args.cache_dir}_splits"
        
        # Split input file
        split_files = split_jsonl_file(args.input, split_dir, args.num_splits)
        
        # Process each split file
        output_files = []
        for i, split_file in enumerate(split_files):
            output_file = os.path.join(split_dir, f"output_{i:03d}.jsonl")
            output_files.append(output_file)
            
            print(f"Processing split {i+1}/{len(split_files)}: {split_file}")
            process_jsonl_with_frame_cache_multithread(
                input_jsonl_path=split_file,
                output_jsonl_path=output_file,
                cache_dir=args.cache_dir,
                fps=args.fps,
                max_workers=args.workers,
                use_processes=not args.use_threads
            )
        
        # Merge output files
        merge_jsonl_files(output_files, args.output)
        
    else:
        # Single file mode with multiprocessing
        video_cache_mapping = process_jsonl_with_frame_cache_multithread(
            input_jsonl_path=args.input,
            output_jsonl_path=args.output,
            cache_dir=args.cache_dir,
            fps=args.fps,
            max_workers=args.workers,
            use_processes=not args.use_threads
        )
    
    # Validate results
    validate_processed_jsonl(args.output)