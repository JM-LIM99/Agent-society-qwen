# Agent Society — Report

Paper: `2605.20179v1.pdf`

Wall-clock: multi 321.9s, baseline 26.7s

## Judge verdict
```json
{
  "A": {
    "completeness": 4,
    "faithfulness": 3,
    "design_depth": 4
  },
  "B": {
    "completeness": 5,
    "faithfulness": 5,
    "design_depth": 5
  },
  "winner": "B",
  "_multi_is": "B"
}
```

## Multi-agent design
# **TIDE: Efficient and Lossless MoE Diffusion LLM Inference with I/O-aware Expert Offload**

## **1. Problem Statement**

The paper addresses the challenge of efficient inference for **MoE-based diffusion large language models (dLLMs)** on **resource-constrained systems**. The key challenges are:

- **High memory footprint**: MoE dLLMs such as **LLaDA2.0-mini** (16B parameters) and **LLaDA2.0-flash** (100B parameters) use **256 FFN experts** with a **top-k = 8** activation pattern.
- **Fragmented expert footprints**: Due to the parallel nature of diffusion decoding, all experts may be activated simultaneously within a block, leading to **out-of-memory (OOM) errors**.
- **I/O overhead from expert swapping**: Each denoising step activates a larger and more diverse set of experts than an AR step, resulting in **massive CPU-GPU data transfers**.
- **CPU-bound execution**: As more tokens are routed to the host, the GPU becomes idle while waiting for CPU-processed activations.

These limitations hinder the performance of MoE-dLLM inference on resource-constrained platforms.

---

## **2. Core Components of TIDE**

### **2.1 Interval-Based Expert Refresh Strategy**

- **Refresh steps** (e.g., $ t_0 $ or $ t_\tau $): Experts are dynamically updated based on **token hit counts**.
- **Skipped steps** ($ t_1 $ to $ t_{\tau-1} $): The system continues decoding with the current expert placement without migration.
- This strategy minimizes unnecessary expert migrations and optimizes **GPU utilization**.

### **2.2 Asynchronous Execution Pipeline**

- When a token is routed to a CPU expert ("miss"), the GPU does not stall; instead, the token features are offloaded to the CPU for concurrent processing while the GPU continues executing "hits" for other tokens in the batch.
- Results are re-synchronized at the end of the FFN block, effectively overlapping **slower CPU computation** with **high-throughput GPU execution**.

### **2.3 Lossless Inference**

- TIDE focuses on **expert placement** without modifying the selection of the MoE router or model weights.
- Since each token is assigned to the same set of experts as in GPU-only execution, and the parallel decoding mechanism is preserved, the model outputs remain unchanged.
- Therefore, TIDE is a **lossless optimization** that provides **free-lunch style acceleration** for MoE-dLLM inference.

### **2.4 Mathematical Programming for Optimal Interval Selection**

- The paper formulates an optimization problem to determine the optimal **refresh interval $ \tau $** that balances I/O overhead and compute efficiency.
- The objective function is:
  $$
  \min_{\tau \in [1, 2, ..., T-1]} \left( \frac{B \cdot T}{\tau} \cdot (1 - (1 - d)^{\tau}) + C_{\text{CPU}} \cdot T \cdot B \cdot f(\tau) \right)
  $$
  where:
  - $ B $: Number of GPU experts
  - $ T $: Number of denoising steps
  - $ d $: Drift rate (approximation of cross-step similarity)
  - $ C_{\text{CPU}} $: CPU-related computation constant
  - $ f(\tau) $: Monotonically increasing function of $ \tau $

---

## **3. Data Flow of TIDE**

### **3.1 Overall Process**

1. **Initialization**:
   - The system initializes the full expert set $ E $, dividing it into CPU and GPU expert sets $ E_{\text{CPU}} $ and $ E_{\text{GPU}} $.
   - The number of GPU experts $ B $ is fixed based on available memory.

2. **Token Routing and Expert Placement**:
   - At **refresh steps** (e.g., $ t_0 $ or $ t_\tau $), the system updates the GPU-resident expert set by promoting 'high-demand' experts with the highest token hits from CPU memory to GPU memory, while evicting 'low-demand' experts back to CPU memory.
   - During **skipped steps**, the system maintains a fixed expert placement and routes tokens to their corresponding expert sets in an asynchronous fashion.

3. **Asynchronous Execution**:
   - When a token is routed to a CPU expert ("miss"), the GPU does not stall; instead, the token features are offloaded to the CPU for concurrent processing while the GPU continues executing "hits" for other tokens in the batch.
   - Results are re-synchronized at the end of the FFN block.

4. **Performance Optimization**:
   - The optimal refresh interval $ \tau $ is determined by solving a constrained mathematical programming problem using hardware profiling and greedy search.
   - This ensures minimal I/O overhead and maximal compute efficiency.

---

## **4. Performance Evaluation**

### **4.1 Throughput Improvements**

- **LLaDA2.0-mini**:
  - Under **GPU memory constraint of 10 GB**, TIDE achieves **2.11 tokens/s** vs. **1.81 tokens/s** for Fiddler and **1.69 tokens/s** for Mixtral-Offload.
  - With **18 GB GPU memory**, TIDE reaches **2.36 tokens/s**, outperforming both baselines.
- **LLaDA2.0-flash**:
  - Under **30 GB GPU memory**, TIDE achieves **1.25 tokens/s**, surpassing Fiddler (**1.01 tokens/s**) and Mixtral-Offload (**1.01 tokens/s**).
- **Overall Speedup**:
  - TIDE achieves up to **1.4× and 1.5× speedups** over prior baselines for LLaDA2.0-mini and LLaDA2.0-flash, respectively.

### **4.2 Impact of Refresh Interval $ \tau $**

- For **LLaDA2.0-mini** with a **block size of 32** and **GPU expert budget of 64**, TIDE achieves **1.89 tokens/s** with optimal $ \tau $, compared to **1.62 tokens/s** with random $ \tau $ and **1.79 tokens/s** with $ \tau=1 $.
- This demonstrates that **interval-based expert refresh** effectively balances I/O overhead and compute efficiency.

### **4.3 Performance Across Configurations**

- TIDE consistently outperforms all baseline methods regardless of decoding settings.
- It maintains **robust performance improvements** even when GPU memory is constrained.
- TIDE also sustains consistent improvement across different confidence thresholds, achieving an average **1.4× speedup** over prior baselines.

---

## **5. Implementation Details**

- **Hardware Configuration**:
  - **LLaDA2.0-mini** is run on an **NVIDIA A100 40 GB GPU**.
  - **LLaDA2.0-flash** is run on an **NVIDIA H100 80 GB GPU**.
  - System uses a **48-core Intel CPU** and **1024 GB DDR4 host memory**.
- **Implementation Tools**:
  - TIDE is implemented using **HuggingFace Transformers** and **dInfer**, with **PyTorch 2.9** and **CUDA 12.8**.
- **Datasets**:
  - Experiments are conducted on the **sanitized MBPP dataset** sourced from the `lm_eval_harness` library.

---

## **6. Summary of Core Contributions**

| Contribution | Description |
|--------------|-------------|
| **Identification of Challenges** | Highlights the limitations of existing MoE inference methods and the need for new approaches. |
| **Insight into Temporal Stability** | Leverages the fact that expert activation patterns remain stable across steps, enabling intelligent expert placement. |
| **Formulation of Optimization Problem** | Models latency overheads and solves a constrained mathematical programming problem to determine the optimal refresh interval. |
| **Interval-Based Expert Refresh Strategy** | Reduces unnecessary expert migrations and optimizes GPU utilization. |
| **Asynchronous Execution Pipeline** | Overlaps CPU and GPU computations to minimize idle time. |
| **Lossless Inference** | Provides free-lunch style acceleration without modifying model weights or router decisions. |
| **Performance Evaluation** | Demonstrates up to **1.4× and 1.5× throughput improvements** over prior baselines on LLaDA2.0-mini and LLaDA2.0-flash models. |

These contributions establish TIDE as a promising solution for efficient and lossless inference of MoE-based diffusion language models on resource-constrained systems.

# CODE
```python
import torch
from typing import Dict, List, Set, Optional, Tuple, Any
from abc import ABC, abstractmethod
import math


class TIDEExpertManager:
    """
    Implements TIDE: Efficient and Lossless MoE Diffusion LLM Inference with I/O-aware Expert Offload
    
    This class manages expert placement between CPU and GPU memories with interval-based refresh
    and asynchronous execution pipeline to optimize MoE-dLLM inference on resource-constrained systems.
    
    Based on the paper's core components:
    - Interval-Based Expert Refresh Strategy
    - Asynchronous Execution Pipeline  
    - Lossless Inference guarantees
    - Mathematical Programming for Optimal Interval Selection
    """
    
    def __init__(self, 
                 total_experts: int,
                 gpu_expert_budget: int,
                 num_denoising_steps: int,
                 drift_rate: float,  # 'd' in the optimization formula - approximation of cross-step similarity
                 cpu_compute_constant: float,  # C_CPU in the optimization formula
                 refresh_interval: Optional[int] = None,  # tau in the paper - if None, will be computed
                 device_memory_limit: Optional[float] = None):  # not specified in source
        """
        Initialize the TIDE expert manager.
        
        Args:
            total_experts: Total number of experts in the MoE model (e.g., 256 FFN experts)
            gpu_expert_budget: Number of GPU experts B - fixed based on available memory
            num_denoising_steps: Number of denoising steps T in the diffusion process
            drift_rate: Drift rate 'd' - approximation of cross-step similarity
            cpu_compute_constant: C_CPU - CPU-related computation constant
            refresh_interval: Refresh interval tau - if None, will be optimized
            device_memory_limit: Memory limit of the GPU device [not specified in summary]
        """
        self.total_experts = total_experts
        self.gpu_expert_budget = gpu_expert_budget
        self.num_denoising_steps = num_denoising_steps
        self.drift_rate = drift_rate
        self.cpu_compute_constant = cpu_compute_constant
        self.device_memory_limit = device_memory_limit
        
        # Initialize expert sets
        self.gpu_experts: Set[int] = set()
        self.cpu_experts: Set[int] = set(range(total_experts))
        
        # Token hit counters for expert usage tracking
        self.token_hit_counts: Dict[int, int] = {i: 0 for i in range(total_experts)}
        
        # Determine optimal refresh interval if not provided
        if refresh_interval is None:
            self.refresh_interval = self._compute_optimal_refresh_interval()
        else:
            self.refresh_interval = refresh_interval
            
        self.current_step = 0
        
    def _compute_optimal_refresh_interval(self) -> int:
        """
        Mathematical Programming for Optimal Interval Selection
        Solves: min_tau (B*T/tau * (1-(1-d)^tau) + C_CPU*T*B*f(tau))
        where f(tau) is a monotonically increasing function of tau
        """
        # The paper doesn't specify the exact form of f(tau), so we'll assume f(tau) = tau
        # This is a simplification based on the description that f(tau) is monotonically increasing
        best_tau = 1
        best_cost = float('inf')
        
        for tau in range(1, self.num_denoising_steps):
            # Cost function from the paper: B*T/tau * (1-(1-d)^tau) + C_CPU*T*B*f(tau)
            io_overhead = (self.gpu_expert_budget * self.num_denoising_steps / tau) * \
                         (1 - (1 - self.drift_rate) ** tau)
            cpu_compute_overhead = self.cpu_compute_constant * self.num_denoising_steps * \
                                  self.gpu_expert_budget * tau
            
            total_cost = io_overhead + cpu_compute_overhead
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_tau = tau
                
        return best_tau
    
    def is_refresh_step(self) -> bool:
        """
        Interval-Based Expert Refresh Strategy
        Determines if the current step is a refresh step (t_0 or t_tau) vs skipped steps (t_1 to t_{tau-1})
        """
        return self.current_step % self.refresh_interval == 0
    
    def update_expert_placement(self):
        """
        Interval-Based Expert Refresh Strategy
        Updates GPU-resident expert set by promoting high-demand experts from CPU to GPU
        and evicting low-demand experts from GPU to CPU
        """
        # Get experts sorted by hit count (descending order)
        sorted_experts = sorted(
            self.token_hit_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Select top-k experts for GPU based on hit counts
        selected_gpu_experts = {exp_id for exp_id, _ in sorted_experts[:self.gpu_expert_budget]}
        
        # Update expert sets
        self.gpu_experts = selected_gpu_experts
        self.cpu_experts = set(range(self.total_experts)) - selected_gpu_experts
        
        # Reset hit counts after refresh
        self.token_hit_counts = {i: 0 for i in range(self.total_experts)}
    
    def route_tokens_to_experts(self, token_ids: List[int], expert_assignments: List[List[int]]) -> Tuple[
        List[Tuple[int, List[int]]],  # GPU tokens and their expert assignments
        List[Tuple[int, List[int]]]   # CPU tokens and their expert assignments
    ]:
        """
        Token Routing and Expert Placement with Asynchronous Execution Pipeline
        Routes tokens to their corresponding expert sets (GPU or CPU) based on current placement
        
        Args:
            token_ids: List of token IDs in the current batch
            expert_assignments: List of expert IDs assigned to each token (top-k selection)
            
        Returns:
            Tuple of (gpu_routed_tokens, cpu_routed_tokens) with their respective expert assignments
        """
        gpu_routed = []
        cpu_routed = []
        
        # Count token hits for each expert to inform future refresh decisions
        for token_idx, experts_for_token in enumerate(expert_assignments):
            for expert_id in experts_for_token:
                self.token_hit_counts[expert_id] += 1
        
        for token_idx, experts_for_token in enumerate(expert_assignments):
            token_id = token_ids[token_idx]
            
            # Check if ALL experts for this token are on GPU (hit) or some are on CPU (miss)
            all_on_gpu = all(expert_id in self.gpu_experts for expert_id in experts_for_token)
            
            if all_on_gpu:
                gpu_routed.append((token_id, experts_for_token))
            else:
                cpu_routed.append((token_id, experts_for_token))
                
        return gpu_routed, cpu_routed
    
    def execute_async_forward_pass(self, 
                                 gpu_routed_tokens: List[Tuple[int, List[int]]],
                                 cpu_routed_tokens: List[Tuple[int, List[int]]],
                                 expert_modules: Dict[int, torch.nn.Module]) -> Dict[int, torch.Tensor]:
        """
        Asynchronous Execution Pipeline
        Executes GPU computations while offloading CPU computations concurrently
        Results are re-synchronized at the end of the FFN block
        
        Args:
            gpu_routed_tokens: Tokens assigned to GPU experts
            cpu_routed_tokens: Tokens assigned to CPU experts  
            expert_modules: Dictionary mapping expert IDs to their PyTorch modules
            
        Returns:
            Dictionary mapping token IDs to their computed output tensors
        """
        results = {}
        
        # Execute GPU computations asynchronously
        if gpu_routed_tokens:
            # Process all GPU tokens in parallel on GPU
            gpu_results = self._execute_gpu_computation(gpu_routed_tokens, expert_modules)
            results.update(gpu_results)
        
        # Execute CPU computations (asynchronously in real implementation)
        if cpu_routed_tokens:
            cpu_results = self._execute_cpu_computation(cpu_routed_tokens, expert_modules)
            results.update(cpu_results)
            
        return results
    
    def _execute_gpu_computation(self, 
                               gpu_routed_tokens: List[Tuple[int, List[int]]],
                               expert_modules: Dict[int, torch.nn.Module]) -> Dict[int, torch.Tensor]:
        """
        Execute computations for tokens assigned to GPU-resident experts
        """
        results = {}
        
        for token_id, expert_list in gpu_routed_tokens:
            # Simulate token computation through its assigned experts
            # In real implementation, this would involve actual tensor operations
            output_tensor = torch.randn(1, 512).cuda()  # Placeholder for actual computation
            results[token_id] = output_tensor
            
        return results
    
    def _execute_cpu_computation(self, 
                               cpu_routed_tokens: List[Tuple[int, List[int]]],
                               expert_modules: Dict[int, torch.nn.Module]) -> Dict[int, torch.Tensor]:
        """
        Execute computations for tokens assigned to CPU-resident experts
        In real implementation, this runs concurrently with GPU computation
        """
        results = {}
        
        for token_id, expert_list in cpu_routed_tokens:
            # Simulate token computation through its assigned experts on CPU
            # In real implementation, this would involve actual tensor operations
            output_tensor = torch.randn(1, 512)  # Placeholder for actual computation
            results[token_id] = output_tensor
            
        return results
    
    def step(self, 
             token_ids: List[int], 
             expert_assignments: List[List[int]],
             expert_modules: Dict[int, torch.nn.Module]) -> Dict[int, torch.Tensor]:
        """
        Main execution step that implements the complete TIDE workflow
        
        Args:
            token_ids: Current batch of token IDs
            expert_assignments: Top-k expert assignments for each token
            expert_modules: Dictionary of expert neural network modules
            
        Returns:
            Dictionary mapping token IDs to their computed outputs
        """
        # Check if this is a refresh step
        if self.is_refresh_step():
            # Update expert placement based on token hit counts
            self.update_expert_placement()
        
        # Route tokens to appropriate expert sets (GPU or CPU)
        gpu_routed, cpu_routed = self.route_tokens_to_experts(token_ids, expert_assignments)
        
        # Execute forward pass with asynchronous pipeline
        results = self.execute_async_forward_pass(gpu_routed, cpu_routed, expert_modules)
        
        # Increment step counter
        self.current_step += 1
        
        return results


class TIDEMoELayer(torch.nn.Module):
    """
    A sample MoE layer implementation that works with TIDE's expert management system
    """
    
    def __init__(self, 
                 num_experts: int, 
                 expert_hidden_size: int, 
                 top_k: int = 8,  # e.g., top-k = 8 activation pattern from paper
                 **kwargs):
        """
        Initialize the MoE layer compatible with TIDE
        
        Args:
            num_experts: Total number of experts in the MoE (e.g., 256 from paper)
            expert_hidden_size: Hidden dimension size for each expert
            top_k: Number of experts to activate per token (top-k selection)
        """
        super().__init__()
        self.num_experts = num_experts
        self.expert_hidden_size = expert_hidden_size
        self.top_k = top_k
        
        # Create expert modules (these would typically be stored separately for TIDE)
        self.experts = torch.nn.ModuleList([
            torch.nn.Linear(expert_hidden_size, expert_hidden_size)
            for _ in range(num_experts)
        ])
        
        # Router network for expert selection
        self.router = torch.nn.Linear(expert_hidden_size, num_experts)
        
    def forward(self, 
                hidden_states: torch.Tensor, 
                expert_manager: TIDEExpertManager,
                **kwargs) -> torch.Tensor:
        """
        Forward pass that integrates with TIDE's expert routing and management
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            expert_manager: TIDEExpertManager instance for expert placement decisions
            
        Returns:
            Output hidden states after MoE processing
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_dim)
        
        # Get router logits for expert selection
        router_logits = self.router(hidden_flat)  # [batch_size*seq_len, num_experts]
        
        # Get top-k expert assignments for each token
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        
        # Convert to lists for TIDE expert manager
        token_ids = list(range(hidden_flat.size(0)))
        expert_assignments = [indices.tolist() for indices in top_k_indices]
        
        # Use TIDE expert manager to route tokens and execute computations
        results = expert_manager.step(token_ids, expert_assignments, 
                                    {i: self.experts[i] for i in range(self.num_experts)})
        
        # Combine results (simplified - in practice this involves more complex aggregation)
        output = torch.zeros_like(hidden_flat)
        for token_id, result_tensor in results.items():
            output[token_id] = result_tensor.squeeze()[:hidden_dim]
        
        return output.view(batch_size, seq_len, hidden_dim)
```


## Baseline
### Summary of Core Contributions

**1. Introduction of TIDE: A Resource-Efficient Inference System for MoE-dLLMs**
- **Core Idea:** TIDE is a novel inference system that optimizes the resource usage of diffusion-based large language models (dLLMs) with mixture-of-experts (MoE) architectures.
- **Key Innovation:** It leverages the temporal stability of expert activations during the diffusion process, enabling intelligent expert placement and reducing I/O overhead.

**2. Interval-Based Expert Refresh Strategy**
- **Mechanism:** TIDE introduces an interval-based refresh strategy to update expert placements in an I/O-aware manner.
- **Optimization:** The optimal interval is determined by solving a constrained mathematical programming problem, minimizing both I/O traffic and CPU computation.

**3. Lossless Optimization**
- **No Model Training Required:** TIDE does not require any model training, making it a "free lunch" type of acceleration.
- **Preservation of Accuracy:** It preserves model accuracy while significantly improving inference throughput.

**4. Implementation and Evaluation**
- **Evaluation on LLaDA2.0 Models:** TIDE achieves up to 1.4× and 1.5× throughput improvements over prior baselines on LLaDA2.0-mini and LLaDA2.0-flash models, respectively.
- **Cross-Platform Compatibility:** Implemented on top of HuggingFace Transformers and dInfer, with support for PyTorch 2.9 and CUDA 12.8.

---

### System Design for TIDE

#### 1. Overview
TIDE is designed as an I/O-aware inference system for MoE-dLLMs, aiming to improve system throughput by intelligently managing expert placement and token routing.

#### 2. Key Components

##### 2.1 Expert Refresh Strategy
- **Interval-Based Refresh:** Experts are refreshed at regular intervals (`τ`), where `τ` is determined through an optimization process.
- **Expert Migration:** During refresh steps, experts with high token hit counts are promoted from CPU to GPU memory, while low-demand experts are evicted back to CPU memory.

##### 2.2 Token Routing
- **Asynchronous Processing:** Tokens are routed to their respective experts asynchronously. If the expert is on the GPU, processing continues on the GPU; if the expert is on the CPU, processing is offloaded to the CPU.
- **Efficient Execution:** This ensures that the GPU remains busy, maximizing compute efficiency and minimizing idle time.

##### 2.3 Mathematical Programming for Interval Selection
- **Optimization Problem:** TIDE formulates the interval selection as a constrained mathematical programming problem, using an analytical model to minimize total latency.
- **Hardware Profiling:** Hardware profiling is used to estimate constants related to CPU computation and I/O bandwidth, aiding in the optimization process.

#### 3. Algorithmic Details

##### 3.1 TIDE Scheduling Policy
- **Algorithm Overview:** The algorithm dynamically updates expert placements at regular intervals (`τ`) and routes tokens to their corresponding experts.
- **Steps:**
  1. At each decoding step, check if it's a refresh step.
  2. Update expert placements by promoting high-demand experts to GPU and evicting low-demand ones to CPU.
  3. Route tokens to their respective experts, performing asynchronous processing on both GPU and CPU.
  4. Re-synchronize results at the end of the FFN block to ensure consistency.

##### 3.2 Asynchronous Execution Pipeline
- **Token Offloading:** When a token is routed to a CPU-resident expert, the GPU continues processing other tokens.
- **Overlap of Operations:** The slower CPU computations are overlapped with the faster GPU execution, improving overall throughput.

#### 4. Performance Metrics
- **Throughput (token/s):** Measured as the number of tokens decoded per second.
- **I/O Overhead:** Minimized by leveraging the temporal stability of expert activation patterns.
- **CPU Utilization:** Maintained at high levels by efficiently managing expert migration and token routing.

#### 5. Evaluation Results
- **Performance Comparison:** TIDE outperforms existing methods such as Fiddler and Mixtral-Offload across various configurations.
- **Scalability:** Demonstrates consistent performance improvements across different block sizes and GPU budgets.
- **Lossless Inference:** Preserves model accuracy while achieving significant speedups.

---

### Conclusion

TIDE represents a significant advancement in the efficient inference of MoE-dLLMs by leveraging the temporal stability of expert activations. Through an interval-based expert refresh strategy and asynchronous execution pipeline, TIDE minimizes I/O overhead and maximizes compute efficiency, offering a lossless and resource-efficient solution for diffusion-based large language models.