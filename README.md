# Qwen3-ASR 电影字幕生成

本项目用于给完整电影生成 SRT 字幕，基于 Qwen3-ASR-1.7B。

## 环境准备

```bash
conda create -n qwen-asr python=3.12 -y
conda activate qwen-asr

# transformers 后端
pip install -U qwen-asr

# 可选：vLLM 后端（更快）
# pip install -U qwen-asr[vllm]

# 可选：FlashAttention 2（GPU 支持时可加速/降显存）
# pip install -U flash-attn --no-build-isolation
```

系统需要安装 ffmpeg。

## GPU 环境信息（示例）

```
NVIDIA-SMI 580.95.05              Driver Version: 580.95.05      CUDA Version: 13.0
GPU: NVIDIA GB10
```

## 使用方法

```bash
python scripts/movie_subtitle.py \
  --input /path/to/movie.mp4 \
  --output /path/to/output.srt \
  --segment-seconds 300 \
  --backend transformers
```

使用 vLLM：

```bash
python scripts/movie_subtitle.py \
  --input /path/to/movie.mp4 \
  --output /path/to/output.srt \
  --segment-seconds 300 \
  --backend vllm
```

## 使用本地模型（已下载到本地）

如果模型已下载在 `~/Desktop/model/modelscope/models`，可以直接传本地路径（注意本机目录名不含 `Qwen/` 前缀）：

```bash
python scripts/movie_subtitle.py \
  --input /path/movie.mp4 \
  --output /path/output.srt \
  --segment-seconds 300 \
  --model /path/models/Qwen3-ASR-1.7B \
  --forced-aligner /path/models/Qwen3-ForcedAligner-0.6B \
  --backend vllm
```
## 断点续跑 / 失败重试

- 断点续跑：加 `--resume`，会从 `--tmp` 目录里的 `progress.json` 继续。
- 重试次数：`--retries N`（默认 2）
- 重试间隔：`--retry-backoff 秒`（默认 5 秒）

示例：

```bash
python scripts/movie_subtitle.py \
  --input /path/to/movie.mp4 \
  --output /path/to/output.srt \
  --segment-seconds 300 \
  --resume \
  --retries 3 \
  --retry-backoff 10
```

## 说明

- `--segment-seconds 300`：强制对齐最长 5 分钟，建议保持 ≤300 秒。
- `--no-timestamps`：关闭强制对齐，速度更快但时间轴更粗。
- `--max-new-tokens`：长片段可能需要更大值。
- `--language`：强制指定识别语言（如 `Chinese` 或 `English`），不传则自动识别。
