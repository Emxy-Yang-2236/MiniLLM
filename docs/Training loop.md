# Training Loop

We now finally put together the major components built so far: the tokenized data, the model, and the optimizer.

## Data Loader

The tokenized data (e.g., that you prepared in `tokenizer_experiments`) is a single sequence of tokens $x = (x_1, \ldots, x_n)$. 
Even though the source data might consist of separate documents (e.g., different web pages, or source code files), 
a common practice is to concatenate all of those into a single sequence of tokens, adding a delimiter between them (such as the `<|endoftext|>` token).

A *data loader* turns this into a stream of *batches*, where each batch consists of $B$ sequences of length $m$, paired with the corresponding next tokens, also with length $m$. 
For example, for $B = 1$, $m = 3$, $([x_2, x_3, x_4], [x_3, x_4, x_5])$ would be one potential batch.

Loading data in this way simplifies training for a number of reasons. 
First, any $1 \leq i \leq n - m$ gives a valid training sequence, so sampling training sequences is trivial. 
Since all training sequences have the same length, there’s no need to *pad* input sequences, which improves hardware utilization (also by increasing batch size $B$). 
Finally, we also don’t need to load the full dataset to sample training data, making it easy to handle large datasets that might not otherwise fit in memory.


Write a function that takes a numpy array $x$ (integer array with token IDs), a `batch_size`, a `context_length` and a PyTorch device string (e.g., `'cpu'` or `'cuda:0'`), and returns a pair of tensors: 
the sampled input sequences and the corresponding next-token targets. 
Both tensors should have shape `(batch_size, context_length)` containing token IDs, and both should be placed on the requested device. 

> Your implementation should be in `release/minillm/data/pretrain_dataset.py` (def get_batch).

> You can ignore the complex hardware and system issues mentioned below; `get_batch` is merely an exemplary task and will not be invoked in the final pipeline. (You only need to pass the tests.)

> **Low-Resource Tip:** Data loading on CPU or Apple Silicon
>
> ---
>
> If you are planning to train your LM on CPU or Apple Silicon, you need to move your data to the correct device (and similarly, you should use the same device for your model later on).
>
> If you are on CPU, you can use the `'cpu'` device string, and on Apple Silicon (M* chips), you can use the `'mps'` device string.
>
> For more on MPS, check out these resources:
>
> - https://docs.pytorch.org/docs/stable/mps.html
> - https://docs.pytorch.org/docs/stable/notes/mps.html
> - https://developer.apple.com/documentation/metalperformanceshaders

What if the dataset is too big to load into memory? 
We can use a Unix system call named `mmap` which maps a file on disk to virtual memory, and lazily loads the file contents when that memory location is accessed. 
Thus, you can “pretend” you have the entire dataset in memory. Numpy implements this through `np.memmap` (or the flag `mmap_mode='r'` to `np.load`, if you originally saved the array with `np.save`), which will return a numpy array-like object that loads the entries on-demand as you access them. 
**When sampling from your dataset (i.e., a numpy array) during training, be sure to load the dataset in memory-mapped mode** (via `np.memmap` or the flag `mmap_mode='r'` to `np.load`, depending on how you saved the array). 
Make sure you also specify a `dtype` that matches the array that you’re loading. 
It may be helpful to explicitly verify that the memory-mapped data looks correct (e.g., doesn’t contain values beyond the expected vocabulary size).


## Checkpointing

In addition to loading data, we will also need to save models as we train. 
When running jobs, we often want to be able to resume a training run that stopped midway through (e.g., due to your job timing out, machine failure, etc). 
Even when all goes well, we might also want to later have access to intermediate models (e.g., to study training dynamics post-hoc, take samples from models at different stages of training, etc).

A checkpoint should have all the states that we need to resume training. 
We of course want to be able to restore model weights at a minimum. 
If using a stateful optimizer (such as AdamW), we will also need to save the optimizer’s state (e.g., in the case of AdamW, the moment estimates). 
Finally, to resume the learning rate schedule, we will need to know the iteration number we stopped at. 
PyTorch makes it easy to save all of these: every `nn.Module` has a `state_dict()` method that returns a dictionary with all learnable weights; 
we can restore these weights later with the sister method `load_state_dict()`. 
The same goes for any `torch.optim.Optimizer`. 
Finally, `torch.save(obj, dest)` can dump an object (e.g., a dictionary containing tensors as some values, but also regular Python objects like integers) to a file (path) or file-like object, which can then be loaded back into memory with `torch.load(src)`.

Implement the following two functions to load and save checkpoints:

`def save_checkpoint(model, optimizer, iteration, out)` should dump all the state from the model, optimizer and iteration into the file-like object `out`. 
You can use the `state_dict` method of both the model and the optimizer to get their relevant states and use `torch.save(obj, out)` to dump `obj` into `out` (PyTorch supports either a path or a file-like object here). 
A typical choice is to have `obj` be a dictionary, but you can use whatever format you want as long as you can load your checkpoint later.

This function expects the following parameters:
- `model: torch.nn.Module`
- `optimizer: torch.optim.Optimizer`
- `iteration: int`
- `out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]`

`def load_checkpoint(src, model, optimizer)` should load a checkpoint from `src` (path or file-like object), and then recover the model and optimizer states from that checkpoint. 
Your function should return the iteration number that was saved to the checkpoint. 
You can use `torch.load(src)` to recover what you saved in your `save_checkpoint` implementation, and the `load_state_dict` method in both the model and optimizer to return them to their previous states.

This function expects the following parameters:
- `src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]`
- `model: torch.nn.Module`
- `optimizer: torch.optim.Optimizer`

> Your implementation should be in `release/minillm/train/checkpoint.py`.


## Training loop

Now, it’s finally time to put all of the components you implemented together into your main training script. 
MiniLLM provides the orchestration for configuration, tokenizer/model construction, data loaders, checkpoint file names, metrics files, and sample generation.
Your job is to fill in the core training logic, similar to the SGD example in the optimizer section:

```python
optimizer.zero_grad()
loss = model(input_ids, labels)["loss"]
loss.backward()
clip_grad_norm_(model.parameters(), max_norm)
optimizer.step()
```

In `release/minillm/train/pretrain.py`, complete:

- `evaluate_loss(...)`: switch the model to eval mode, disable gradients, run validation batches, and return average loss.
- `train_one_step(...)`: set the learning rate, fetch training batches, compute loss, backpropagate, clip gradients, step the optimizer and scheduler, update `TrainState`, and return a metrics row.

The provided `train_pretrain(cfg, max_steps=None)` wrapper calls these functions and handles the surrounding project pipeline.

## Running TinyStories Pretraining

After the tokenizer, Transformer LM, optimizer, scheduler, checkpointing, and generation tests pass, run pretraining from `release/`.
Pretraining uses encoded `.bin` files, not raw `.txt` files.

### Smoke run: quick correctness check

```bash
cd release
python scripts/run_student_pipeline.py --mode smoke --device cpu
```

Smoke mode is only a path check. It should produce:

```text
runs/student_pipeline/smoke/pretrain/checkpoint_last.pt
outputs/smoke/metrics_pretrain.jsonl
outputs/smoke/pretrain_samples.md
```

### Full run: main training evidence

```bash
cd release
python scripts/run_student_pipeline.py --mode student --device cuda
```

Use `--device mps` on Apple Silicon or `--device cpu` if CUDA is unavailable.
This is the main run for the final report. Inspect:

```text
runs/student_pipeline/student/pretrain/checkpoint_last.pt
outputs/release_candidate/metrics_pretrain.jsonl
outputs/release_candidate/pretrain_samples.md
outputs/release_candidate/run_summary.md
```

The pretraining loss should move downward, and fixed-prompt samples should become more TinyStories-like than random text.

### Manual debug: pretraining only

If tokenizer and encoded `.bin` files already exist, you can debug only the pretraining loop:

```bash
cd release
python scripts/train_pretrain.py \
  --config configs/pretrain_smoke.yaml \
  --max_steps 4 \
  --device cpu
```

For full pretraining only:

```bash
cd release
python scripts/train_pretrain.py \
  --config configs/pretrain_student.yaml \
  --device cuda
```

Manual `train_pretrain.py` does not run SFT, final evaluation, or report generation.
Use `run_student_pipeline.py --mode student` when you need final assignment artifacts.
