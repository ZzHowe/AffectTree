# Dependency Installation
Please install the corresponding dependencies according to the EasyR1 codebase.
Additionally, run:
```bash
./examples/prepare_env_AffectTree.sh
```

# Data Preparation
For different datasets, please prepare video frames and corresponding jsonl datasets separately.
The jsonl dataset items should include both `video_path` and `frame_cache_path` fields.

Specifically, `frame_cache_path` should be generated using:
```bash
./verl/utils/prepare_frames.py
```

For the input jsonl file (containing the video path field), output a jsonl file containing `frame_cache_path`. You need to specify the frame cache directory and the fps for frame extraction, which is uniformly set to 8:
```bash
python prepare_frames.py \
    --input your_raw_data_jsonl \
    --output final_data_jsonl \
    --cache-dir your_frames_cache_dir \
    --fps 8
```

Note: Please update `./examples/run_AffectTree.sh` according to the processed `final_data_jsonl`.

# Run
```bash
bash ./examples/run_AffectTree.sh
```