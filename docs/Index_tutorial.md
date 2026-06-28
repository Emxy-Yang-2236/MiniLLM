# Index

- Deep learning introduction: [intro.md](./intro.md)  
    **If you are not in a rush, this section is highly recommended.**  
    Covered:  
    - What is deep learning
    - Basic components of learning settings
    - 3 examples: linear regression, logic regression and perceptron
    - Activations
    - Neural network and why it can fit functions
    - Training introduction
    - Generalization

- (Week1) Tokenizer: [BPE tokenizer.md](./BPE%20tokenizer.md)  
    Covered:  
    - Why we need a tokenizer
    - BPE tokenizer
    - How to train a BPE tokenizer
    - Tokenizer encoder and decoder

- (Week1 & Week2) 
    1. Transformer LM: [Transformer LM.md](./Transformer%20LM.md)  
        Covered  
    2. Training a Transformer LM: [Training a Transformer LM.md](./Training%20a%20Transformer%20LM.md)
        Covered:  
        - cross-entropy
        - SGD and AdamW
        - learning-rate scheduling
        - gradient clipping
    3. Training loop: [Training loop.md](./Training%20loop.md)
        Covered:
        - data loading
        - checkpointing
        - putting the training loop together
    4. Generating text: [Generating text.md](./Generating%20text.md)
        Covered:
        - softmax decoding
        - temperature
        - top-p sampling

- (Week3)
    1. Tiny SFT: [Tiny SFT.md](./Tiny%20sft.md)  
        Covered:
        - what supervised fine-tuning means in this project
        - prompt/response data
        - SFT loss
        - SFT training flow
        - before/after samples and held-out eval

- (Week4)
    1. Training Measurement Mini-lab: [Training Measurement Mini-lab.md](./Training%20Measurement%20Mini-lab.md)  
        Covered:
        - what a training-step benchmark measures
        - naive attention vs SDPA
        - fp32 vs bf16/fp16 when available
        - sequence length and batch-size trends
        - benchmark limitations
