# Large Language Models Notes

## Basics
- Linear algebra
- Calculus 
- Probability theory 
- Statistics
- Computer sciences basics

## Python (modern)
- with type hints https://github.com/panaverse/learn-modern-python
- project and package (and virtual env) management https://github.com/astral-sh/uv 
- modern practices https://www.stuartellis.name/articles/python-modern-practices/

## Generative engineering
- https://github.com/panaverse/learn-generative-ai

## Tools 
### Running models
- Huggingface: site/libs
- aichat vs aider vs goose
- llama.cpp vs ollama vs vllam

### Editor support
- llama vim plugin
- 
### Benchmarking
- llama bench 

## Hardware
### M1 Ultra 
Your M1 Ultra with **120GB RAM** is a powerhouse for running LLMs, especially if optimized correctly. Here’s how you can get the most out of it:  

---

### **1. Use Metal Backend for GPU Acceleration**
Apple Silicon's **Metal Performance Shaders (MPS)** can speed up inference for LLMs like Llama, Mistral, and Deepseek.  

- Install PyTorch with Metal support:  
  ```bash
  pip install torch torchvision torchaudio
  ```
- Run models with `device="mps"` in PyTorch.  

---

### **2. Choose the Right Model Format**
Since you have **120GB RAM**, you can run **larger models efficiently**. Some options:  

- **GGUF Models (best for Apple Silicon)**
  - Download optimized GGUF models from [TheBloke on Hugging Face](https://huggingface.co/TheBloke)
  - Use `llama.cpp`:  
    ```bash
    git clone https://github.com/ggerganov/llama.cpp  
    cd llama.cpp && make  
    ./main -m model.gguf -p "Hello, world!"
    ```
  
- **MLX (Apple’s native ML framework)**
  - **Optimized for M-series GPUs**  
  - Run Llama or Mistral models using [mlx-lm](https://github.com/ml-explore/mlx-lm):  
    ```bash
    pip install mlx  
    git clone https://github.com/ml-explore/mlx-lm  
    ```

---

### **3. Quantization for Speed & Efficiency**
For larger models, quantization reduces memory usage while maintaining performance.  
- **Use 4-bit or 5-bit quantized models** (e.g., `Q4_K_M`, `Q5_K_M` in GGUF)  
- If using `transformers`, try `bitsandbytes` for 4-bit inference:
  ```bash
  pip install bitsandbytes
  ```

---

### **4. Run Local Servers for Faster Inference**
Instead of CLI tools, you can host your LLM with **Ollama** or **LM Studio** for a smoother experience.
- **Ollama** (easy to use, optimized for macOS)  
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh  
  ollama run mistral  
  ```
- **LM Studio** (GUI-based, runs GGUF models efficiently)

---

### **5. Optimize Memory & CPU Usage**
- **Use `taskset` to pin processes** to performance cores (`p-cores`)  
- **Monitor RAM & GPU usage**  
  ```bash
  vm_stat  
  powermetrics --samplers gpu_power -i 1000  
  ```
- **Run models in batch mode** for efficient token generation.  

---

### **Final Thoughts**  
Your M1 Ultra + 120GB RAM is **great for running 30B+ models** at high speed. I'd recommend trying **Ollama for convenience**, **mlx for Apple-optimized inference**, and **GGUF models for best efficiency**.  

Your M1 Ultra with 120GB RAM is a powerhouse for running LLMs locally. Here’s how you can optimize it for **Ollama** or **llama.cpp** to get the best performance:

---

### **1. Use Metal Acceleration (GPU)**
Both **Ollama** and **llama.cpp** support **Apple’s Metal backend**, which lets you run models on the GPU instead of just the CPU. This is **crucial** for speed.

#### ✅ **For `llama.cpp`**
Run with Metal acceleration:
```bash
./main -m model.gguf --n-gpu-layers 100 --threads 10
```
- `--n-gpu-layers 100`: Load as many layers as possible into the GPU (tune based on VRAM availability).
- `--threads 10`: Use multiple CPU threads to balance workload.

#### ✅ **For `Ollama`**
Ollama uses Metal by default, but you can check with:
```bash
OLLAMA_METAL=1 ollama run <model>
```

---

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

