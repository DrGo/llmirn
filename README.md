# Large Language Models Notes

## Basics
- Linear algebra
- Calculus 
- Probability theory 
- Statistics
- Computer sciences basics

## Quantization 

## Python (modern)
- with type hints https://github.com/panaverse/learn-modern-python
- project and package (and virtual env) management https://github.com/astral-sh/uv 
- modern practices https://www.stuartellis.name/articles/python-modern-practices/

## Go (as alternative to python)
- gomlx: https://eli.thegreenplace.net/2024/gomlx-ml-in-go-without-python/
https://github.com/gomlx/gomlx
- //TODO build a Go Api for Apple MLX https://github.com/ml-explore/mlx/issues/60

## Generative engineering
- https://github.com/panaverse/learn-generative-ai
Curriculum 
https://github.com/jacobhilton/deep_learning_curriculum/blob/master/1-Transformers.md

Implementing a transformer
https://nlp.seas.harvard.edu/annotated-transformer/

Google Colab   
Llama3.1_(8B)-Alpaca.ipynb: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-Alpaca.ipynb

## Models
### Deepseek-r1 
- quantized https://unsloth.ai/blog/deepseekr1-dynamic#running%20r1

For Apple Metal devices, be careful of --n-gpu-layers. If you find the machine going out of memory, reduce it. For a 128GB unified memory machine, you should be able to offload 59 layers or so.

./llama.cpp/llama-cli \
    --model DeepSeek-R1-GGUF/DeepSeek-R1-UD-IQ1_S/DeepSeek-R1-UD-IQ1_S-00001-of-00003.gguf \
    --cache-type-k q4_0 \
    --threads 16 \
    --prio 2 \
    --temp 0.6 \
    --ctx-size 8192 \
    --seed 3407 \
    --n-gpu-layers 59 \
    -no-cnv \
    --prompt "<｜User｜>Create a Flappy Bird game in Python.<｜Assistant｜>"

## Tools 
### Running models
- Huggingface: site/libs
- aichat vs aider vs goose
- llama.cpp vs ollama vs vllam

#### llama.cpp 


### Editor support
- llama vim plugin
- 
### Benchmarking
- llama bench 

## Hardware
### Nvidia 
- https://2080ti22g.com/blogs/news/putting-together-a-rig-with-8-x-rtx-2080ti22g 

### AMD
https://videocardz.com/newz/amd-ryzen-ai-max-300-strix-halo-apus-bring-up-to-16-zen5-cores-and-40-rdna3-5-compute-units

### M1 Ultra 
### Optimize for llm
1. Use Metal Backend for GPU Acceleration
Apple Silicon's **Metal Performance Shaders (MPS)** can speed up inference for LLMs 
- Run models with `device="mps"` in PyTorch.  
2. Choose the Right Model Format**
- **GGUF Models (best for Apple Silicon)**
  - Download optimized GGUF models from [TheBloke on Hugging Face](https://huggingface.co/TheBloke)
- **MLX (Apple’s native ML framework)** **Optimized for M-series GPUs**  
  - Run Llama or Mistral models using [mlx-lm](https://github.com/ml-explore/mlx-lm):  
3. Quantization for Speed & Efficiency**
For larger models, quantization reduces memory usage while maintaining performance.  
- **Use 4-bit or 5-bit quantized models** (e.g., `Q4_K_M`, `Q5_K_M` in GGUF)  
- If using `transformers`, try `bitsandbytes` for 4-bit inference:
  ```bash
  pip install bitsandbytes
  ```

5. Optimize Memory & CPU Usage**
- **Use `taskset` to pin processes** to performance cores (`p-cores`)  
- **Monitor RAM & GPU usage**  
  ```bash
  vm_stat  
  powermetrics --samplers gpu_power -i 1000  
  ```
- **Run models in batch mode** for efficient token generation.  

1. Use Metal Acceleration (GPU)**
```bash
./main -m model.gguf --n-gpu-layers 100 --threads 10
```
- `--n-gpu-layers 100`: Load as many layers as possible into the GPU (tune based on VRAM availability).
- `--threads 10`: Use multiple CPU threads to balance workload.

Ollama uses Metal by default, but you can check with:
```bash
OLLAMA_METAL=1 ollama run <model>
```

### **2. Pick the Right Model Format**
For best performance:
- **Use GGUF models** for llama.cpp (better Apple Silicon optimization).
- **Quantized models (Q4, Q5, Q6, or GGUF-K) balance performance and accuracy**.
  - Try **Q5_K_M** for a mix of speed and accuracy.
  - Use **Q3_K_M** if you need to fit larger models.

Download quantized models from [TheBloke’s Hugging Face repo](https://huggingface.co/TheBloke).

Example:
```bash
ollama pull llama3:8b-q5_K_M
```

---

### **3. Tune CPU & Memory Usage**
Your M1 Ultra has **20 CPU cores** (16 performance + 4 efficiency), so you should:
- **Use 10–16 threads for llama.cpp** (`--threads 10`)
- **Allocate more context length cautiously** (higher means more RAM usage):
  ```bash
  ./main -m model.gguf -c 4096 --n-gpu-layers 100
  ```
  (Avoid going beyond 8192 unless necessary.)

- **Monitor RAM & Swap usage:**
  ```bash
  vm_stat | grep "free"
  ```

If you hit swap, **lower the context size (`-c 2048`) or reduce `--n-gpu-layers`**.

---

### **4. Use Swap on NVMe SSD (If Needed)**
If running **large models (e.g., 65B)**, you may need virtual memory:
```bash
sudo sysctl -w vm.swapusage
```
macOS manages swap, but you can create a manual swapfile for better control.

---

### **5. Use a Fast Disk (If Loading Models Often)**
If you **frequently switch models**, store them on an **NVMe external SSD (Thunderbolt)** for faster loading.

---

### **6. Experiment with Alternative Runtimes**
Besides **llama.cpp and Ollama**, you might also check:
- **Mistral (via Ollama)**: `ollama pull mistral`
- **vLLM on macOS (WIP)**: Lower memory footprint for bigger models.

---

### **7. Benchmark Performance**
Run tests to see if optimizations help:
```bash
./main -m model.gguf -p "Test prompt" -t 10 --n-gpu-layers 50
```
Or:
```bash
ollama benchmark
```

---

### **Conclusion**
With your M1 Ultra:
- Use **Metal acceleration** for GPU speed.
- Pick **GGUF quantized models** (Q5_K_M for balance).
- Tune `--n-gpu-layers` and `--threads` for efficiency.
- Monitor RAM & swap to avoid slowdowns.

### Monitor gpu/cpu/mem use 
- brew install macmon https://github.com/vladkens/macmon

## Benchmarks
https://llm.aidatatools.com/results-macos.php 


## Configuration
I think most of the model creators share their model usage examples so high at 0.6-0.7 simply because it's what a lot of the client apps use. IMO this is WAY too high unless you're doing creative writing.
Generally I set temp to 0-0.4 at absolute most.
min_p actually needs a little temperature to work effectively so with min_p I almost always use 0.2
