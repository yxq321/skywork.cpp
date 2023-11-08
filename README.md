# skywork.cpp | 天工大模型通过CPU来运行

- 基于 [llama.cpp](https://github.com/ggerganov/llama.cpp) 通过 C/C++ 来实现的大模型运行环境，可以通过 CPU 就可以直接运行 [天工大模型](https://github.com/SkyworkAI/Skywork)。

## 使用说明
### 克隆本仓库
```bash
git clone https://github.com/yxq321/skywork.cpp
cd skywork.cpp
```

### 安装相关依赖并编译
在 Linux 或 MacOS上，运行`make`:
```bash
make
```
### 从Huggingface上下载大模型

运行`download-skywork.py`下载天工大模型，默认存放在`  ~/.cache/huggingface/hub/` 目录下
```bash
skywork.cpp# python3 download-skywork.py
Downloading (…)bcec297346/README.md: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 21.8k/21.8k [00:00<00:00, 110MB/s]
Downloading (…)5%8D%8F%E8%AE%AE.pdf: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 266k/266k [00:00<00:00, 32.3MB/s]
Downloading (…)iguration_skywork.py: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 3.12k/3.12k [00:00<00:00, 22.2MB/s]
Downloading (…)neration_config.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 182/182 [00:00<00:00, 1.79MB/s]
Downloading (…)ec297346/config.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 733/733 [00:00<00:00, 6.77MB/s]
Downloading (…)sc/skywork_logo.jpeg: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 78.9k/78.9k [00:00<00:00, 55.9MB/s]
Downloading chat_demo_1.gif: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2.37M/2.37M [00:00<00:00, 205MB/s]
Downloading chat_demo_2.gif: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 108k/108k [00:00<00:00, 46.2MB/s]
Downloading (…)isc/skywork_icon.png: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 8.10k/8.10k [00:00<00:00, 48.7MB/s]
Downloading (…)97346/.gitattributes: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 5.46k/5.46k [00:00<00:00, 39.0MB/s]
Downloading chat_demo_3.gif: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 556k/556k [00:00<00:00, 51.1MB/s]
Downloading (…)c/stage1_metrics.png: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 270k/270k [00:00<00:00, 87.1MB/s]
Downloading (…)isc/stage2_ceval.png: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 128k/128k [00:00<00:00, 66.0MB/s]
Downloading (…)sc/training_loss.png: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 30.7k/30.7k [00:00<00:00, 52.7MB/s]
...
Downloading (…)okenizer_config.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 857/857 [00:00<00:00, 5.30MB/s]
Downloading (…)l-00049-of-00053.bin: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 510M/510M [00:02<00:00, 235MB/s]
Downloading (…)l-00051-of-00053.bin: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 510M/510M [00:02<00:00, 217MB/s]
Downloading (…)l-00046-of-00053.bin: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 510M/510M [00:05<00:00, 88.1MB/s]
Downloading (…)l-00050-of-00053.bin: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 510M/510M [00:04<00:00, 111MB/s]
Downloading (…)l-00052-of-00053.bin: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 510M/510M [00:04<00:00, 116MB/s]
Downloading (…)l-00053-of-00053.bin: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.21G/1.21G [00:05<00:00, 203MB/s]
```
### 转换成GGUF格式

参照下面运行 `convert-skywork-hf-to-gguf.py`，请替换成本机实际路径:

```bash
skywork.cpp# python3 convert-skywork-hf-to-gguf.py  ~/.cache/huggingface/hub/models--Skywork--Skywork-13B-Base/snapshots/2f15ad62f302f9e0015ec941dd4eeabcec297346/ 1 --outfile=skywork-f16.gguf
gguf: Conversion Endianess 0
gguf: loading model 2f15ad62f302f9e0015ec941dd4eeabcec297346
hello print:  SkyworkForCausalLM
gguf: found 53 model parts
num_parts:53

This gguf file is for Little Endian only
gguf: get model metadata
gguf: get tokenizer metadata
gguf: get sentencepiece tokenizer vocab, scores and token types
gguf: Setting special token type bos to 1
gguf: Setting special token type eos to 2
gguf: Setting special token type pad to 0
gguf: get tensor metadata
gguf: loading model part 'pytorch_model-00001-of-00053.bin'
model.layers.0.input_layernorm.weight -> blk.0.attn_norm.weight, n_dims = 1, torch.bfloat16 --> float32
model.layers.0.post_attention_layernorm.weight -> blk.0.ffn_norm.weight, n_dims = 1, torch.bfloat16 --> float32
model.layers.0.self_attn.q_proj.weight -> blk.0.attn_q.weight, n_dims = 2, torch.bfloat16 --> float16
model.layers.0.self_attn.k_proj.weight -> blk.0.attn_k.weight, n_dims = 2, torch.bfloat16 --> float16
model.layers.0.self_attn.v_proj.weight -> blk.0.attn_v.weight, n_dims = 2, torch.bfloat16 --> float16
model.layers.0.self_attn.o_proj.weight -> blk.0.attn_output.weight, n_dims = 2, torch.bfloat16 --> float16
...
model.layers.51.self_attn.k_proj.weight -> blk.51.attn_k.weight, n_dims = 2, torch.bfloat16 --> float16
model.layers.51.self_attn.v_proj.weight -> blk.51.attn_v.weight, n_dims = 2, torch.bfloat16 --> float16
model.layers.51.self_attn.o_proj.weight -> blk.51.attn_output.weight, n_dims = 2, torch.bfloat16 --> float16
model.layers.51.mlp.gate_proj.weight -> blk.51.ffn_gate.weight, n_dims = 2, torch.bfloat16 --> float16
model.layers.51.mlp.up_proj.weight -> blk.51.ffn_up.weight, n_dims = 2, torch.bfloat16 --> float16
model.layers.51.mlp.down_proj.weight -> blk.51.ffn_down.weight, n_dims = 2, torch.bfloat16 --> float16
gguf: loading model part 'pytorch_model-00053-of-00053.bin'
model.norm.weight -> output_norm.weight, n_dims = 1, torch.bfloat16 --> float32
model.embed_tokens.weight -> token_embd.weight, n_dims = 2, torch.bfloat16 --> float16
lm_head.weight -> output.weight, n_dims = 2, torch.bfloat16 --> float16
gguf: write header
gguf: write metadata
gguf: write tensors
gguf: model successfully exported to 'skywork-f16.gguf'
```
### (可选)量化

如果本机内存不够，可以在上面基础上进一步量化，由16位量化成4位，方法如下:
```bash
skywork.cpp# ./quantize skywork-f16.gguf skywork-q4_0.gguf q4_0
main: build = 1498 (e980088)
main: built with cc (Ubuntu 5.4.0-6ubuntu1~16.04.12) 5.4.0 20160609 for x86_64-linux-gnu
main: quantizing 'skywork-f16.gguf' to 'skywork-q4_0.gguf' as Q4_0
llama_model_loader: loaded meta data with 18 key-value pairs and 471 tensors from skywork-f16.gguf (version GGUF V3 (latest))
llama_model_loader: - tensor    0:           blk.0.attn_norm.weight f32      [  4608,     1,     1,     1 ]
llama_model_loader: - tensor    1:            blk.0.ffn_norm.weight f32      [  4608,     1,     1,     1 ]
llama_model_loader: - tensor    2:              blk.0.attn_q.weight f16      [  4608,  4608,     1,     1 ]
llama_model_loader: - tensor    3:              blk.0.attn_k.weight f16      [  4608,  4608,     1,     1 ]
llama_model_loader: - tensor    4:              blk.0.attn_v.weight f16      [  4608,  4608,     1,     1 ]
llama_model_loader: - tensor    5:         blk.0.attn_output.weight f16      [  4608,  4608,     1,     1 ]
llama_model_loader: - tensor    6:            blk.0.ffn_gate.weight f16      [  4608, 12288,     1,     1 ]
[ 468/ 471]               blk.51.ffn_down.weight - [12288,  4608,     1,     1], type =    f16, quantizing to q4_0 .. size =   108.00 MB ->    30.38 MB | hist: 0.036 0.015 0.024 0.037 0.055 0.076 0.097 0.115 0.122 0.115 0.097 0.076 0.055 0.037 0.024 0.020
[ 469/ 471]                   output_norm.weight - [ 4608,     1,     1,     1], type =    f32, size =    0.018 MB
[ 470/ 471]                    token_embd.weight - [ 4608, 65519,     1,     1], type =    f16, quantizing to q4_0 .. size =   575.85 MB ->   161.96 MB | hist: 0.036 0.015 0.025 0.038 0.056 0.077 0.097 0.112 0.118 0.112 0.097 0.077 0.056 0.038 0.025 0.021
[ 471/ 471]                        output.weight - [ 4608, 65519,     1,     1], type =    f16, quantizing to q6_K .. size =   575.85 MB ->   236.19 MB | hist:
llama_model_quantize_internal: model size  = 26425.55 MB
llama_model_quantize_internal: quant size  =  7507.74 MB
llama_model_quantize_internal: hist: 0.036 0.016 0.025 0.039 0.056 0.077 0.096 0.112 0.118 0.112 0.096 0.077 0.056 0.039 0.025 0.021

main: quantize time = 43989.90 ms
main:    total time = 43989.90 ms
skywork.cpp# ls -lh *.gguf
-rw-r--r-- 1 root root  26G Nov  8 07:23 skywork-f16.gguf
-rw-r--r-- 1 root root 7.4G Nov  8 08:05 skywork-q4_0.gguf
```

## 运行
```bash
skywork.cpp# ./main -m skywork-f16.gguf -p "陕西的省会是西安"
Log start
main: build = 1498 (e980088)
main: built with cc (Ubuntu 5.4.0-6ubuntu1~16.04.12) 5.4.0 20160609 for x86_64-linux-gnu
main: seed  = 1699428993
llama_model_loader: loaded meta data with 18 key-value pairs and 471 tensors from skywork-f16.gguf (version GGUF V3 (latest))
llama_model_loader: - tensor    0:           blk.0.attn_norm.weight f32      [  4608,     1,     1,     1 ]
llama_model_loader: - tensor    1:            blk.0.ffn_norm.weight f32      [  4608,     1,     1,     1 ]
llama_model_loader: - tensor    2:              blk.0.attn_q.weight f16      [  4608,  4608,     1,     1 ]
llama_model_loader: - tensor    3:              blk.0.attn_k.weight f16      [  4608,  4608,     1,     1 ]
llama_model_loader: - tensor    4:              blk.0.attn_v.weight f16      [  4608,  4608,     1,     1 ]
...
llama_model_loader: - type  f32:  105 tensors
llama_model_loader: - type  f16:  366 tensors
llm_load_vocab: mismatch in special tokens definition ( 1847/65519 vs 259/65519 ).
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = skywork
llm_load_print_meta: vocab type       = SPM
llm_load_print_meta: n_vocab          = 65519
llm_load_print_meta: n_merges         = 0
llm_load_print_meta: n_ctx_train      = 131072
llm_load_print_meta: n_embd           = 4608
llm_load_print_meta: n_head           = 36
llm_load_print_meta: n_head_kv        = 36
llm_load_print_meta: n_layer          = 52
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_gqa            = 1
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: n_ff             = 12288
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 10000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_yarn_orig_ctx  = 131072
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: model type       = 13B
llm_load_print_meta: model ftype      = mostly F16 (guessed)
llm_load_print_meta: model params     = 13.85 B
llm_load_print_meta: model size       = 25.81 GiB (16.00 BPW)
llm_load_print_meta: general.name   = 2f15ad62f302f9e0015ec941dd4eeabcec297346
llm_load_print_meta: BOS token = 1 '<s>'
llm_load_print_meta: EOS token = 2 '</s>'
llm_load_print_meta: UNK token = 0 '<unk>'
llm_load_print_meta: PAD token = 0 '<unk>'
llm_load_print_meta: LF token  = 13 '<0x0A>'
llm_load_tensors: ggml ctx size =    0.17 MB
llm_load_tensors: mem required  = 26425.72 MB
.................................................................................................
llama_new_context_with_model: n_ctx      = 512
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 1
llama_new_context_with_model: kv self size  =  468.00 MB
llama_build_graph: non-view tensors processed: 1096/1096
llama_new_context_with_model: compute buffer total size = 143.60 MB

system_info: n_threads = 20 / 40 | AVX = 1 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | FMA = 1 | NEON = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS
= 0 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 |
sampling:
        repeat_last_n = 64, repeat_penalty = 1.100, frequency_penalty = 0.000, presence_penalty = 0.000
        top_k = 40, tfs_z = 1.000, top_p = 0.950, min_p = 0.050, typical_p = 1.000, temp = 0.800
        mirostat = 0, mirostat_lr = 0.100, mirostat_ent = 5.000
generate: n_ctx = 512, n_batch = 512, n_predict = -1, n_keep = 0


陕西的省会是西安，在古代就是著名的长安城，位于现在的关中地区。西安市，也被誉为十三朝古都。西安，秦始皇兵马俑坑，是世界八大奇迹之一。
所以，秦朝时期是我国历史上第一个大一统封建王朝。西北的首都所在。秦朝、唐朝、宋朝等都属于中原文化区。

公元前221年，唐太宗李世民的长安城，唐长安是当时世界上最大的城市，它是唐朝都城长安是当时亚洲首屈一指的商业中心，拥有人口最多最繁华的国际贸易都市，是丝绸之路起点！
西安人吃“十三绝”
唐代长安城的中轴线朱雀大街就是今天的解放路。【西安钟楼、西安城墙、西安交通枢纽、西安钟楼、西安钟楼、西安钟楼，以西安古为背景，这个地方应该有很多历史遗迹、文化古迹和博物馆，是西安最繁华地段，是游客最多的地方
所以在晚上最热闹了，现在已经不是旅游景点，但依然是西安最古老的一条街。
中山路，西边就是以前的街道，东边是城墙，这里非常适合散步，而到回民区！
陕西省博物馆是一定要去看看。西安这座城市的标志性建筑之一。2013年才开放，在历史上有着十分重要的地位，有很多人在那里的地方，是由很多的，而且还有很多的景点都可以参观，
这里有非常多的人文景观，很值得来打卡拍照呢。
```
