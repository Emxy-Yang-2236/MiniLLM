# Byte-Pair Encoding (BPE) Tokenizer

## Why Do We Need a Tokenizer?

Before a neural network can process text, we need to convert text into numbers. 
A neural network does not directly understand strings such as "Once upon a time"; it operates on tensors of integers and floating-point numbers.
So the tokenizer is the **first step** in a language modeling pipeline.

In this project, the tokenizer sits before the neural network. 
Tokenizer is not a part of neural network, but every later component depends on it.
Language models receive token IDs, look up their embeddings, and predict the next token.

So before we build the Transformer, we first build the tokenizer.

## What will you do?

In week 1, we will train and implement a byte-level byte-pair encoding (BPE) tokenizer [1, 2]. 
In particular, we will represent arbitrary (Unicode) strings as a sequence of bytes and train our BPE tokenizer on this byte sequence. 
Later, we will use this tokenizer to encode text (a string) into tokens (a sequence of integers) for language modeling.

Now, let's get into the world of tokenizer.

## The Unicode Standard

Unicode is a text encoding standard that maps characters to integer code points. As of Unicode 17.0 (released in September 2025), the standard defines 159,801 characters across 172 scripts. 
For example, the character “s” has the code point 115 (typically notated as U+0073, where U+ is a conventional prefix and 0073 is 115 in hexadecimal), and the character “牛” has the code point 29275. 
In Python, you can use the `ord()` function to convert a single Unicode character into its integer representation. 
The `chr()` function converts an integer Unicode code point into a string with the corresponding character.

```python
>>> ord('牛')
29275
>>> chr(29275)
'牛'
```


## Unicode Encodings

While the Unicode standard defines a mapping from characters to code points (integers), it’s impractical to train tokenizers directly on Unicode code points, since the vocabulary would be prohibitively large (around 150K items) and sparse (since many characters are quite rare). 
Instead, we’ll use a Unicode encoding, which converts a Unicode character into a sequence of bytes. 
The Unicode standard itself defines three encodings: UTF-8, UTF-16, and UTF-32, with UTF-8 being the dominant encoding for the Internet (more than 98% of all webpages).

To encode a Unicode string into UTF-8, we can use the `encode()` function in Python. 
To access the underlying byte values for a Python **bytes** object, we can iterate over it (e.g., call `list()`). 
Finally, we can use the `decode()` function to decode a UTF-8 byte string into a Unicode string.

```python
>>> test_string = "hello! こんにちは!"
>>> utf8_encoded = test_string.encode("utf-8")
>>> print(utf8_encoded)
b'hello! \xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf!'
>>> print(type(utf8_encoded))
<class 'bytes'>
>>> # Get the byte values for the encoded string (integers from 0 to 255).
>>> list(utf8_encoded)
[104, 101, 108, 108, 111, 33, 32, 227, 129, 147, 227, 130, 147, 227, 129, 171, 227, 129,
161, 227, 129, 175, 33]
>>> # One byte does not necessarily correspond to one Unicode character!
>>> print(len(test_string))
>>> print(len(utf8_encoded))
>>> print(utf8_encoded.decode("utf-8"))
hello! こんにちは!
```

By converting our Unicode code points into a sequence of bytes (e.g., via the UTF-8 encoding), we are essentially taking a sequence of code points (21-bit integers with 159,801 valid values) and transforming it
into a sequence of byte values (integers in the range 0 to 255). 
The 256-length byte vocabulary is much more manageable to deal with. 
When using byte-level tokenization, we do not need to worry about out-of-vocabulary tokens, since we know that any input text can be expressed as a sequence of integers from 0 to 255.


## Subword Tokenization

While byte-level tokenization can alleviate the out-of-vocabulary issues faced by word-level tokenizers, tokenizing text into bytes results in extremely long input sequences. 
This slows down model training, since a sentence with 10 words might only be 10 tokens long in a word-level language model, but could be 50 or more tokens long in a character-level model (depending on the length of the words). 
Processing these longer sequences requires more computation at each step of the model. 
Furthermore, language modeling on byte sequences is difficult because the longer input sequences create long-term dependencies in the data.

Subword tokenization is a midpoint between word-level tokenizers and byte-level tokenizers. 
Note that a byte-level tokenizer’s vocabulary has 256 entries (byte values are 0 to 255). 
A subword tokenizer trades off a larger vocabulary size for better compression of the input byte sequence. 
For example, if the byte sequence `b'the'` often occurs in our raw text training data, assigning it an entry in the vocabulary would reduce this 3-token sequence to a single token.

How do we select these subword units to add to our vocabulary? Sennrich et al. [1] propose to use byte-pair encoding (BPE; Gage [3]), a compression algorithm that iteratively replaces (“merges”) the most frequent pair of bytes with a single, new unused index. 
Note that this algorithm adds subword tokens to our vocabulary to maximize the compression of our input sequences —  
if a word occurs in our input text enough times, it’ll be represented as a single subword unit.

Subword tokenizers with vocabularies constructed via BPE are often called BPE tokenizers. 
We’ll implement a byte-level BPE tokenizer, where the vocabulary items are bytes or merged sequences of bytes, which give us the best of both worlds in terms of out-of-vocabulary handling and manageable input sequence lengths. 
The process of constructing the BPE tokenizer vocabulary is known as “training” the BPE tokenizer. (A different "training" from deep learning.)


## BPE Tokenizer Training

The BPE tokenizer training procedure consists of three main steps.

### 1. Vocabulary initialization

The tokenizer vocabulary is a one-to-one mapping from bytestring token to integer ID. 
Since we’re training a byte-level BPE tokenizer, our initial vocabulary is simply the set of all bytes. 
Since there are 256 possible byte values, our initial vocabulary is of size 256.

### 2. Pre-tokenization

Once you have a vocabulary, you could, in principle, count how often bytes occur next to each other in your text and begin merging them starting with the most frequent pair of bytes. 
However, this is quite computationally expensive, since we’d have to take a full pass over the corpus each time we merge. 
In addition, directly merging bytes across the corpus may result in tokens that differ only in punctuation (e.g., dog! vs. dog.). 
These tokens would get completely different token IDs, even though they are likely to have high semantic similarity (since they differ only in punctuation).

To avoid this, we *pre-tokenize* the corpus. 
You can think of this as a coarse-grained tokenization over the corpus that helps us count how often pairs of characters appear. 
For example, the word 'text' might be a pre-token that appears 10 times. 
In this case, when we count how often the characters ‘t’ and ‘e’ appear next to each other, we will see that the word ‘text’ has ‘t’ and ‘e’ adjacent and we can increment their count by 10 instead of looking through the corpus. 
Since we’re training a byte-level BPE model, each pre-token is represented as a sequence of UTF-8 bytes.

The original BPE implementation of Sennrich et al. [1] pre-tokenizes by simply splitting on whitespace (i.e., `s.split(" ")`). 
This method is still found in tokenizers based on SentencePiece (for instance the Llama 1 and 2 tokenizer).

Most modern tokenizers use a regex-based pre-tokenizer, a practice from GPT-2; see Radford et al. [4].
We’ll use a slightly prettier form of the original regex, fetched from
https://github.com/openai/tiktoken/pull/234/files

```python
>>> PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

It may be useful to interactively split some text with this pre-tokenizer to get a better sense of its behavior:

```python
>>> # requires `regex` package
>>> import regex as re
>>> re.findall(PAT, "some text that i'll pre-tokenize")
['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']
```

When using it in your code, however, you should use re.finditer to avoid storing the pre-tokenized words as you construct your mapping from pre-tokens to their counts.

### 3. Compute BPE merges

Now that we’ve converted our input text into pre-tokens and represented each pre-token as a sequence of UTF-8 bytes, we can compute the BPE merges (i.e., train the BPE tokenizer). 
At a high level, the BPE algorithm iteratively counts every pair of bytes and identifies the pair with the highest frequency ("A", "B"). 
Every occurrence of this most frequent pair ("A", "B") is then *merged*, i.e., replaced with a new token "AB". 
This new merged token is added to our vocabulary; as a result, the final vocabulary after BPE training is the size of the initial vocabulary (256 in our case), plus the number of BPE merge operations performed during training. 
For efficiency during BPE training, we do not consider pairs that cross pre-token boundaries. 

```text
Note that the original BPE formulation of Sennrich et al. [1] specifies the inclusion of an end-of-word token.
We do not add an end-of-word-token when training byte-level BPE models because all bytes (including whitespace and punctuation) are included in the model’s vocabulary. 
Since we’re explicitly representing spaces and punctuation, the learned BPE merges will naturally reflect these word boundaries.
```

When computing merges, deterministically break ties in pair frequency by preferring the lexicographically greater pair. 
For example, if the pairs ("A", "B"), ("A", "C"), ("B", "ZZ"), and ("BA", "A") all have the highest frequency, we’d merge ("BA", "A"):

```python
>>> max([("A", "B"), ("A", "C"), ("B", "ZZ"), ("BA", "A")])
('BA', 'A')
```

> Special tokens

Often, some strings (e.g., `<|endoftext|>`) are used to encode metadata (e.g., boundaries between documents). 
When encoding text, it’s often desirable to treat some strings as "special tokens" that should never be split into multiple tokens (i.e., will always be preserved as a single token). 
For example, the end-of-sequence string `<|endoftext|>` should always be preserved as a single token (i.e., a single integer ID), so we know when to stop generating from the language model. 
These special tokens must be added to the vocabulary, so they have a corresponding fixed token ID.

Algorithm 1 of Sennrich et al. [1] contains an inefficient implementation of BPE tokenizer training (essentially following the steps that we outlined above). 
It may be useful to implement and test this function to check your understanding.

#### Example: BPE training example
Here is a stylized example from Sennrich et al. [1]. Consider a corpus consisting of the following text
```text
low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
```

and the vocabulary has a special token `<|endoftext|>`.

1. Vocabulary  
   We initialize our vocabulary with our special token `<|endoftext|>` and the 256 byte values.

2. Pre-tokenization  
   For simplicity and to focus on the merge procedure, we assume in this example that pre-tokenization simply splits on whitespace. 
   When we pre-tokenize and count, we end up with the frequency table.
   ```
   {low: 5, lower: 2, widest: 3, newest: 6}
   ```

   It is convenient to represent this as a ` dict[tuple[bytes, ...], int]`, e.g. {(l,o,w): 5, …}. 
   Note that even a single byte is a `bytes` object in Python. 
   There is no `byte` type in Python to represent a single byte, just as there is no `char` type in Python to represent a single character.

3. Merges  
   We first look at every successive pair of bytes and sum the frequency of the words where they appear {lo: 7, ow: 7, we: 8, er: 2, wi: 3, id: 3, de: 3, es: 9, st: 9, ne: 6, ew: 6}. 
   The pairs ('e', 's') and ('s', 't') are tied, so we take the lexicographically greater pair, ('s', 't'). 
   We would then merge the pre-tokens so that we end up with {(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,e,st): 3, (n,e,w,e,st): 6}.

   In the second round, we see that (e, st) is the most common pair (with a count of 9) and we would merge into {(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,est): 3, (n,e,w,est): 6}. 
   Continuing this, the sequence of merges we get in the end will be `['s t', 'e st', 'o w', 'l ow', 'w est', 'n e', 'ne west', 'w i', 'wi d', 'wid est', 'low e', 'lowe r']`.

   If we take 6 merges, we have `['s t', 'e st', 'o w', 'l ow', 'w est', 'n e']` and our vocabulary elements would be `[<|endoftext|>, [...256 BYTE CHARS], st, est, ow, low, west, ne]`.

   With this vocabulary and set of merges, the word newest would tokenize as `[ne, west]`.


## Experimenting with BPE Tokenizer Training

Let’s train a byte-level BPE tokenizer on the TinyStories dataset. 
Before you start, we recommend taking a look at the TinyStories dataset to get a sense of what’s in the data.

### Parallelizing pre-tokenization

You will find that a major bottleneck is the pre-tokenization step. 
You can speed up pre-tokenization by parallelizing your code with the built-in library multiprocessing. 
Concretely, we recommend that in parallel implementations of pre-tokenization, you chunk the corpus while ensuring your chunk boundaries occur at the beginning of a special token. 
You are free to use the starter code at the following link
verbatim to obtain chunk boundaries, which you can then use to distribute work across your processes:
https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_basics/pretokenization_example.py

This chunking will always be valid, since we never want to merge across document boundaries. 
For the purposes of the assignment, you can always split in this way. 
Don’t worry about the edge case of receiving a very large corpus that does not contain <|endoftext|>.

### Removing special tokens before pre-tokenization

Before running pre-tokenization with the regex pattern (using re.finditer), you should strip out all special tokens from your corpus (or your chunk, if using a parallel implementation). 
Make sure that you split on your special tokens, so that no merging can occur across the text they delimit. 
For example, if you have a corpus (or chunk) like `[Doc 1]<|endoftext|>[Doc 2]`, you should split on the special token `<|endoftext|>`, and pre-tokenize `[Doc 1]` and `[Doc 2]` separately, so that no merging can occur across the document boundary. 
In other words, special tokens define hard segmentation boundaries during training, but they should not themselves contribute to merge counts.
This can be done using `re.split` with `"|".join(special_tokens)` as the delimiter (with careful use of `re.escape` since `|` may occur in the special tokens). 
The test `test_train_bpe_special_tokens` will test for this.

### Optimizing the merging step

The naive implementation of BPE training in the stylized example above is slow because for every merge, it iterates over all byte pairs to identify the most frequent pair. 
However, the only pair counts that change after each merge are those that overlap with the merged pair. 
Thus, BPE training speed can be improved by indexing the counts of all pairs and incrementally updating these counts, rather than explicitly iterating over each pair of bytes to count pair frequencies. 
You can get significant speedups with this caching procedure, though we note that the merging part of BPE training is not parallelizable in Python.

> You can try training on a subset of the whole Tinystories dataset first.


## BPE Tokenizer: Encoding and Decoding

In the previous part of the assignment, we implemented a function to train a BPE tokenizer on input text to obtain a tokenizer vocabulary and a list of BPE merges. 
Now, we will implement a BPE tokenizer that loads a provided vocabulary and list of merges and uses them to encode and decode text to/from token IDs.

## Encoding text

The process of encoding text by BPE mirrors how we train the BPE vocabulary. 
There are a few major steps.

1.  Pre-tokenize. 
    We first pre-tokenize the sequence and represent each pre-token as a sequence of UTF-8 bytes, just as we did in BPE training. 
    We will be merging these bytes within each pre-token into vocabulary elements, handling each pre-token independently (no merges across pre-token boundaries).
2.  Apply the merges.
    We then take the sequence of vocabulary element merges created during
    BPE training, and apply it to our pre-tokens *in the same order of creation*.


#### Example: BPE encoding example

For example, suppose our input string is `'the cat ate'`, our vocabulary is {0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'}, and our learned merges are `[(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]`. 
First, our pre-tokenizer would split this string into `['the', ' cat', ' ate']`. 
Then, we’ll look at each pre-token and apply the BPE merges.

The first pre-token 'the' is initially represented as [b't', b'h', b'e']. 
Looking at our list of merges, we identify the first applicable merge to be (b't', b'h'), and use that to transform the pre-token into [b'th', b'e']. 
Then, we go back to the list of merges and identify the next applicable merge to be (b'th', b'e'), which transforms the pre-token into [b'the'].
Finally, looking back at the list of merges, we see that there are no more that apply to the string (since the entire pre-token has been merged into a single token), so we are done applying the BPE merges. 
The corresponding integer sequence is `[9]`.

Repeating this process for the remaining pre-tokens, we see that the pre-token ' cat' is represented as [b' c', b'a', b't'] after applying the BPE merges, which becomes the integer sequence [7, 1, 5]. 
The final pre-token ' ate' is [b' at', b'e'] after applying the BPE merges, which becomes the integer sequence [10, 3]. 
Thus, the final result of encoding our input string is `[9, 7, 1, 5, 10, 3]`.

### Special tokens

Your tokenizer should be able to properly handle user-defined special tokens when encoding text (provided when constructing the tokenizer).

### Memory considerations

Suppose we want to tokenize a large text file that we cannot fit in memory. 
To efficiently tokenize this large file (or any other stream of data), we need to break it up into manageable chunks and process each chunk in turn, so that the memory complexity is constant as opposed to linear in the size of the text. 
In doing so, we need to make sure that a token doesn’t cross chunk boundaries, else we’ll get a different tokenization than the naive method of tokenizing the entire sequence in-memory.


## Decoding text

To decode a sequence of integer token IDs back to raw text, we can simply look up each ID’s corresponding entries in the vocabulary (a byte sequence), concatenate them together, and then decode the bytes to a Unicode string. 
Note that input IDs are not guaranteed to map to valid Unicode strings (since a user could input any sequence of integer IDs). 
In the case that the input token IDs do not produce a valid Unicode string, you should replace the malformed bytes with the official Unicode replacement character U+FFFD.
The `errors` argument of `bytes.decode` controls how Unicode decoding errors are handled, and using `errors='replace'` will automatically replace malformed data with the replacement marker.


## MiniLLM Starter Interface

The starter file for this part is `release/minillm/tokenizer/bpe.py`.
It already contains the public interface used by the scripts and tests.
Your job is to fill in the algorithm behind that interface.

There are two main deliverables.
The first deliverable is **BPE training**.
Given raw training text, your code should learn:

```python
vocab: dict[int, bytes]
merges: list[tuple[bytes, bytes]]
```

This is handled by `train_bpe(...)` and `train_bpe_from_file(...)`.
You will implement special-token splitting, GPT-2-like pre-tokenization, initial byte vocabulary construction, pair counting, deterministic merge selection, merge application, and tokenizer saving.

The second deliverable is the **encoder and decoder**.
Given a trained `vocab` and `merges`, your tokenizer should encode from given text to tokens and decode from tokens to text.
The encoding-decoding process must be lossless, ensuring the reconstructed text is identical to the original.

This is handled by `ByteBPETokenizer.encode(...)`, `encode_iterable(...)`, and `decode(...)`.
You will implement special-token matching, BPE merge application during encoding, streaming encoding for large files, and UTF-8-safe decoding.

The official MiniLLM tokenizer has only one special token `SPECIAL_TOKENS = ["<|endoftext|>"]`.


### Suggested implementation order

Start by making `ByteBPETokenizer.initial(...)` correct.
The initial vocabulary should contain all special tokens first, followed by the 256 single-byte tokens.
With the default special token, byte value `0` has token id `1`.

Then implement `pretokenize(...)`.
Use the third-party `regex` package, not Python's built-in `re`.

After that, implement `train_bpe(...)`.
A straightforward version can represent each pre-token as a tuple of byte tokens, use a `Counter` to count repeated pre-tokens, and recompute pair counts after each merge.
For the full TinyStories tokenizer, you may want to add the incremental pair-count optimization discussed above (otherwise training can be very slow).

`train_bpe_from_file(...)` is the file-based wrapper used by scripts.
It should read the full file, or a UTF-8-safe prefix when `max_bytes` is set, train the tokenizer, save `tokenizer.json`, and return the tokenizer.
The arguments `num_workers` and `num_chunks` are present so the starter has the same signature as the reference implementation and release scripts.
A correct serial implementation may ignore them.

Once training works, implement the runtime methods: `encode(...)`, `encode_iterable(...)`, and `decode(...)`.
These methods are what the pretraining and SFT code will use after `tokenizer.json` has been saved and loaded.

The starter already provides `save(...)`, `load(...)`, `describe(...)`, and `stable_hash(...)`.
These helpers do not change the BPE algorithm, but they are required for reproducible runs and for checking that encoded `.bin` files match the tokenizer that produced them.
You should understand what they save, but you do not need to reimplement them.

## Training BPE Tokenizer

### Step 0: Download TinyStories

MiniLLM expects the raw TinyStories files under `../data/raw/tinystories/`, relative to `release/`.
Download the official train and validation text files:

```bash
mkdir -p ../data/raw/tinystories

wget -O ../data/raw/tinystories/TinyStoriesV2-GPT4-train.txt \
  https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt

wget -O ../data/raw/tinystories/TinyStoriesV2-GPT4-valid.txt \
  https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
```

You should train the tokenizer only on the train split. 
The validation split is encoded later for language-model validation, but it is not used to learn BPE merges.

### Step 1: Pass the tokenizer tests

Before training a large tokenizer, make sure your implementation passes the tokenizer tests:

```bash
python -m pytest -q ../shared/tests/test_tokenizer*.py
python -m pytest -q ../shared/tests/test_cs336_tokenizer_contract.py
python -m pytest -q ../shared/tests/cs336_a1_exact/test_tokenizer.py
python -m pytest -q ../shared/tests/cs336_a1_exact/test_train_bpe.py
```

### Step 2: Train a small smoke tokenizer

First run the small tokenizer config. 
This uses only a prefix of TinyStories and is meant for debugging, not for the final model:

```bash
python scripts/train_tokenizer.py --config configs/tokenizer_smoke.yaml
```

This writes:

```text
runs/student_pipeline/smoke/tokenizer/tokenizer.json
runs/student_pipeline/smoke/tokenizer/tokenizer_manifest.json
```

Use this smoke run to catch obvious bugs quickly. 
If this command is very slow or fails, do not start the full tokenizer run yet.

### Step 3: Train the official TinyStories 10k tokenizer

After the smoke tokenizer works, train the tokenizer used by the full MiniLLM project:

```bash
python scripts/train_tokenizer.py --config configs/tokenizer_tinystories_10k.yaml
```

This config uses:

```text
input_path: ../data/raw/tinystories/TinyStoriesV2-GPT4-train.txt
vocab_size: 10000
special_tokens: ["<|endoftext|>"]
pretokenizer: gpt2_like
tie_break: max
```

It writes:

```text
runs/tokenizer_tinystories_10k/tokenizer.json
runs/tokenizer_tinystories_10k/tokenizer_manifest.json
```

The manifest records the source path, source hash, vocab size, number of merges, special tokens, pre-tokenizer type, tie-breaking rule, and elapsed time. 
This file is useful for debugging and for checking that later encoded files were produced with the same tokenizer.

### Step 4: Encode TinyStories train and valid

Once `tokenizer.json` exists, encode both TinyStories splits into binary token files:

```bash
python scripts/encode_dataset.py --config configs/encode_tinystories_10k.yaml
```

This writes:

```text
runs/tokenizer_tinystories_10k/tinystories_train.bin
runs/tokenizer_tinystories_10k/tinystories_train.manifest.json
runs/tokenizer_tinystories_10k/tinystories_valid.bin
runs/tokenizer_tinystories_10k/tinystories_valid.manifest.json
```

Because `vocab_size = 10000 < 65536`, the encoded `.bin` files use `uint16`.
The training code casts token IDs to `torch.long` when it builds batches.

The encoded manifests record the tokenizer hash. 
If you change the tokenizer code, special tokens, pre-tokenizer, or vocabulary size, the old `.bin` files and old checkpoints are stale. 
Regenerate the tokenizer, encoded `.bin` files, and any checkpoints before trusting new training results.


## References

[1] R. Sennrich, B. Haddow, and A. Birch, “Neural Machine Translation of Rare Words with Subword Units,” in Proc. of ACL, 2016.

[2] C. Wang, K. Cho, and J. Gu, “Neural Machine Translation with Byte-Level Subwords.” 2019.

[3] P. Gage, “A new algorithm for data compression,” C Users Journal, vol. 12, no. 2, pp. 23–38, Feb. 1994.

[4] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever, “Language Models are Unsupervised Multitask Learners.” 2019.
