# GPT2-Reproduce

## Section 1

### Add GPT-2 model implementation and can load hugging face weights

在这个初始commit中，实现了 GPT-2 模型的基本架构组件：

- 实现了 `CausalSelfAttention` 类，用于处理自注意力机制
  ```python
  class CausalSelfAttention(nn.Module):
      def __init__(self, config):
          super().__init__()
          assert config.n_embd % config.n_head == 0
          # key, query, value projections
          self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
          # output projection
          self.c_proj = nn.Linear(config.n_embd, config.n_embd)
          # regularization
          self.n_head = config.n_head
          self.n_embd = config.n_embd
          # bias
          self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                               .view(1, 1, config.block_size, config.block_size))
  
      def forward(self, x):
          B, T, C = x.size()
          # compute query, key, value matrices
          qkv = self.c_attn(x)
          q, k, v = qkv.split(self.n_embd, dim=2)
          k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
          q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
          v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
          
          att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
          att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
          att = F.softmax(att, dim=-1)
          y = att @ v
          y = y.transpose(1, 2).contiguous().view(B, T, C)
          y = self.c_proj(y)
          return y
  ```

- 实现了 `MLP` 类，作为 Transformer 块中的前馈网络
  ```python
  class MLP(nn.Module):
      def __init__(self, config):
          super().__init__()
          self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
          self.gelu = nn.GELU(approximate="tanh")
          self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
  
      def forward(self, x):
          x = self.c_fc(x)
          x = self.gelu(x)
          x = self.c_proj(x)
          return x
  ```

- 实现了 `Block` 类，作为 Transformer 的基本构建Block
  ```python
  class Block(nn.Module):
  
      def __init__(self, config):
          super().__init__()
          self.ln_1 = nn.LayerNorm(config.n_embd)
          self.attn = CausalSelfAttention(config)
          self.ln_2 = nn.LayerNorm(config.n_embd)
          self.mlp = MLP(config)
  
      def forward(self, x):
          x = x + self.attn(self.ln_1(x))
          x = x + self.mlp(self.ln_2(x))
          return x
  ```

- 定义了 `GPTConfig` 数据类，用于配置模型参数
  ```python
  @dataclass
  class GPTConfig:
      block_size: int = 1024
      vocab_size: int = 50257
      n_layer: int = 12
      n_head: int = 12
      n_embd: int = 768
  ```

- 实现了 `GPT` 类的基本框架，包括从预训练模型加载权重的功能
  ```python
  class GPT(nn.Module):
  
      def __init__(self, config:GPTConfig):
          super().__init__()
          self.config = config
  
          self.transformer = nn.ModuleDict(dict(
              wte = nn.Embedding(config.vocab_size, config.n_embd),
              wpe = nn.Embedding(config.block_size, config.n_embd),
              h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
              ln_f = nn.LayerNorm(config.n_embd),
          ))
  
          self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
          
      @classmethod
      def from_pretrained(cls, model_type):
          """Loads pretrained GPT-2 model weights from huggingface"""
          assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
          from transformers import GPT2LMHeadModel
          print("loading weights from pretrained gpt: %s" % model_type)
  
          # n_layer, n_head and n_embd are determined from model_type
          config_args = {
              'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
              'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
              'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
              'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
          }[model_type]
          config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
          config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
          # create a from-scratch initialized minGPT model
          config = GPTConfig(**config_args)
          model = GPT(config)
          sd = model.state_dict()
          sd_keys = sd.keys()
          sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param
  
          # init a huggingface/transformers model
          model_hf = GPT2LMHeadModel.from_pretrained(model_type)
          sd_hf = model_hf.state_dict()
  
          # copy while ensuring all of the parameters are aligned and match in names and shapes
          sd_keys_hf = sd_hf.keys()
          sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
          sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
          transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
          # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
          # this means that we have to transpose these weights when we import them
          assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
          for k in sd_keys_hf:
              if any(k.endswith(w) for w in transposed):
                  # special treatment for the Conv1D weights we need to transpose
                  assert sd_hf[k].shape[::-1] == sd[k].shape
                  with torch.no_grad():
                      sd[k].copy_(sd_hf[k].t())
              else:
                  # vanilla copy over the other parameters
                  assert sd_hf[k].shape == sd[k].shape
                  with torch.no_grad():
                      sd[k].copy_(sd_hf[k])
  
          return model
  ```

并且尝试使用hf的预训练参数进行成功加载

```python
model = GPT.from_pretrained('gpt2')
print("Successfully loaded pretrained model")
```



### Implement forward pass and text generation for GPT-2 model

这个commit实现了 GPT 模型的前向传播过程：

- 完成了 `forward` 方法，处理输入序列并生成预测
  ```python
  	def forward(self, idx):
          B, T = idx.size()
          assert T <= self.config.block_size
          pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
          pos_emb = self.transformer.wpe(pos)
          tok_emb = self.transformer.wte(idx)
          x = tok_emb + pos_emb
          for block in self.transformer.h:
              x = block(x)
          x = self.transformer.ln_f(x)
          logits = self.lm_head(x)
          return logits
  ```
  
- 实现了自动检测device的逻辑
  ```python
  # attempt to autodetect the device
  device = "cpu"
  if torch.cuda.is_available():
      device = "cuda"
  elif torch.backends.mps.is_available():
      device = "mps"
  print(f"using device: {device}")
  ```

并且成功使用tiktoken来测试运行前向前向传播的效果
```python

num_return_sequences = 5
max_length = 30

model = GPT.from_pretrained('gpt2')
model.eval()
model.to(device)

# prefix tokens
import tiktoken
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Hello, I'm a language model")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens.to(device)

# generate
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)
        logits = logits[:,-1,:]
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indices, -1, ix)
        x = torch.cat((x, xcol), dim=1)

for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)
```



### Enhance forward method in GPT model to support loss calculation and optimize inference

这个commit优化了 GPT 模型的前向传播过程，使其可以进行loss的计算：

- 优化了`forward`方法，使用交叉熵函数进行loss的计算
  ```python
  	def forward(self, idx, targets=None):
          device = idx.device
          b, t = idx.size()
          assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
          pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
  
          # forward the GPT model itself
          tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
          pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
          x = tok_emb + pos_emb
  
          for block in self.transformer.h:
              x = block(x)
          x = self.transformer.ln_f(x)
  
          if targets is not None:
              # if we are given some desired targets also calculate the loss
              logits = self.lm_head(x)
              loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
          else:
              # inference-time mini-optimization: only forward the lm_head on the very last position
              logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
              loss = None
  
          return logits, loss
  ```

- 添加了input.txt数据集([raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt))用于后续训练以及loss的测试

成功尝试对于一个初始的GPT-2模型，计算对于上述数据集的一个data batch的loss
```python
import tiktoken
enc = tiktoken.get_encoding('gpt2')
with open('input.txt', 'r') as f:
    text = f.read()
text = text[:1000]
tokens = enc.encode(text)
B, T = 4, 32
buf = torch.tensor(tokens[:B*T + 1], device=device)
x = buf[:-1].view(B, T)
y = buf[1:].view(B, T)


# model = GPT.from_pretrained('gpt2')
model = GPT(GPTConfig())
model.to(device)
logits, loss = model(x, y)

print(loss)
```



### Add training loop with AdamW optimizer for GPT model

这个commit加入了Optimizer并使用一个for循环进行反向传播更新权重：

- 使用了`AdamW` Optimizer并进行循环训练，观测到loss在下降
  ```python
  optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
  for i in range(50):
      optimizer.zero_grad()
      logits, loss = model(x, y)
      loss.backward()
      optimizer.step()
      print(f"iter {i}: loss {loss.item():.4f}")
  ```

  


### Implement DataLoaderLite for efficient batch loading in GPT-2 training

这个commit新增了DataLoader类来为训练实现更高效的数据加载：

```python

import tiktoken

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # at init load tokens from disk and store them in memory
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B * T + 1]
        x = (buf[:-1].view(B, T))   # inputs
        y = (buf[1:].view(B, T))   # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y

```



### Implement weight sharing between transformer and language model head in GPT-2

这个commit实现了词嵌入层（token embedding matrix）和语言模型输出层（language model head）之间的权重共享，使得GPT-2 模型能够在减少参数量的同时保持或提高性能

这种权重共享技术最早由 Press & Wolf (2017) 在论文《Using the Output Embedding to Improve Language Models》中提出，并在 Transformer 架构中被广泛采用。

```python
# weight sharing scheme
self.transformer.wte.weight = self.lm_head.weight
```



### Refactor GPT model initialization and weight initialization logic

在 GPT-2 模型中，大多数权重使用均值为 0、标准差为 0.02 的正态分布进行初始化：

```python
def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

对于残差的最后一层`c_proj`需要特殊初始化，它解决了深度网络中的一个关键问题：

1. **梯度消失/爆炸问题**：在深度模型中，如果每个残差块都以相同的方差添加其输出，随着层数增加，输出的总方差会线性增长，导致深层网络不稳定。
2. **解决方案**：通过将残差路径末端的权重初始化为较小的值，每个残差块的贡献被缩小，使得无论网络有多深，总方差保持在可控范围内。

## Section 2

### Try mixed precision matmul and optimize training loop with improved batch size and timing metrics

通过设置`torch.set_float32_matmul_precision('high')`使得在训练的矩阵乘法计算中，可以使用tensor float 32精度来去计算。并且在每个训练的loop中加入了时间和吞吐量的指标，可以看到使用tensor float 32 精度来去计算可以获得3倍的速度提升

```python
for i in range(50):
    t0 = time.time()
    optimizer.zero_grad()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000
    print(f"iter {i}: loss {loss.item():.4f}, dt {dt:.2f}ms, tok/s {x.size(0) * x.size(1) / (dt / 1000):.2f}")
```



### Enable bfloat16 precision training for improved performance in the training loop

参考pytorch [Automatic Mixed Precision — PyTorch Tutorials 2.7.0+cu126 documentation](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html#adding-torch-autocast)，通过加入`torch.autocast` context manager来使得在运行过程中应用bfloat16

```python
 		with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)
```


### Enable Torch compilation for improved model performance in training

通过加入`torch.compile(model)`可以将当前的运行代码进行编译并做相关的算子融合等优化，进一步加速训练速度

### Replace attention mechanism with flash attention for improved efficiency

使用flash attention替代现有的attention计算算法，因为attention不能自动进行算子融合，手动进行算子融合来提升性能。

```python
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

### Specify vocab size to 50304

手动调整vocab_size为50304，可以使得vocab_size不再是50257这个比较ugly的数字，变为2的幂次方的数将会使得gpu更容易优化处理，进而有性能的提升
```python
model = GPT(GPTConfig(vocab_size=50304))
```




## Section 3

### Implement dynamic learning rate scheduling with linear warmup and cosine decay

参考原始GPT3文章中的训练方法，需要有线性的warmup和cosine decay，以及对于Adam Optimizer有对应参数的修改(lr=6e-4, betas=(0.9, 0.95), eps=1e-8)

```python
max_lr = 6e-4
min_lr = max_lr * 0.1
warup_step = 10
max_step = 50
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warup_step:
        return max_lr * (it + 1) / warup_step
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_step:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warup_step) / (max_step - warup_step)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# optimize
optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), eps=1e-8)
for step in range(50):
    t0 = time.time()
    optimizer.zero_grad()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000
    print(f"iter {step} | loss {loss.item():.4f} | norm: {norm:.4f} | dt {dt:.2f}ms | tok/s {x.size(0) * x.size(1) / (dt / 1000):.2f}")

```


### Add optimizer configuration method with weight decay support

给GPT类添加新的方法来更方便地设置Optimizer的参数

```python
    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all params requiring grad
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, others won't
        # all weights in matmuls and embeddings will be weight decayed
        # biases and layernorms won't be weight decayed
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_dcay_params = sum(p.numel() for p in decay_params)
        num_ndcay_params = sum(p.numel() for p in nodecay_params)
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")
        # fused make it faster
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
        )
        return optimizer

```


### Implement gradient accumulation and adjust batch size for improved training efficiency

因为一个显存的大小限制，可以使用串行的策略来进行gradient accumulation来模拟真正一个batch的大小。

```python
total_batch_size = 524288 # 2^19, ~0.5M
B = 64 # micro batch size
T = 1024 # sequence length
assert total_batch_size % (B * T) == 0, f"total_batch_size {total_batch_size} must be divisible by B*T {B*T}"
grad_accum_steps = total_batch_size // (B * T)
print(f"total desired batch size: {total_batch_size}")
print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T)
```

```python
for step in range(50):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
    tokens_per_sec = tokens_processed / dt
    print(
        f"iter {step} | loss {loss_accum.item():.4f} | norm: {norm:.4f} | dt {dt:.2f}ms | tok/s {tokens_per_sec:.2f} | ")
```

### Implement distributed data parallelism

可以使用torchrun来进行多设备的数据并行，增加训练速度

```python
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# set up DDP (distributed data parallel)
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(ddp_local_rank)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True

    # attempt to autodetect the device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")
```



### Download dataset and refactor DataLoaderLite to support shard-based token loading and add split parameter

下载huggingface上的数据集，并且进行预处理

```python
"""
FineWeb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python fineweb.py
Will save shards to the local directory "edu_fineweb10B".
"""

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset, load_from_disk # pip install datasets
from tqdm import tqdm # pip install tqdm

# ------------------------------------------
local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e8) # 100M tokens per shard, total of 100 shards

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# download the dataset HuggingFaceFW/fineweb-edu
fw = load_from_disk("/home/sichaohe/codes/GPT2-Reproduce/fineweb_edu_complete")
import os
tiktoken_cache_dir = "/home/sichaohe/codes/GPT2-Reproduce/tiktoken_cache"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
# init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # end of text token
def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot] # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

# tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
nprocs = max(1, os.cpu_count()//2)
with mp.Pool(nprocs) as pool:
    shard_index = 0
    # preallocate buffer to hold current shard
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None
    for tokens in pool.imap(tokenize, fw, chunksize=16):

        # is there enough space in the current shard for the new tokens?
        if token_count + len(tokens) < shard_size:
            # simply append tokens to current shard
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            # update progress bar
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # write the current shard and start a new one
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
            # split the document into whatever fits in this shard; the remainder goes to next one
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            # populate the next shard with the leftovers of the current doc
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens)-remainder

    # write any remaining tokens as the last shard
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])

```




## Section 4