# Generating Text

Now that we can train models, the last piece we need is the ability to generate text from our model. 
Recall that a language model takes in a (possibly batched) integer sequence of length `sequence_length` and produces a matrix of size `(sequence_length, vocab_size)`, where each element of the sequence is a probability distribution predicting the next token after that position. 
We will now write a few functions to turn this into a sampling scheme for new sequences.

**Softmax**

By standard convention, the language model output is the output of the final linear layer (the “logits”) and so we have to turn this into a normalized probability via the *softmax* operation, which we saw earlier in Equation 10.

**Decoding**

To generate text (decode) from our model, we will provide the model with a sequence of prefix tokens (the “prompt”), and ask it to produce a probability distribution over the vocabulary that predicts the next token in the sequence. 
Then, we will sample from this distribution over the vocabulary items to determine the next output token.

Concretely, one step of the decoding process should take in a sequence $x_{1\ldots t}$ and return a token $x_{t+1}$ via the following equation,

$$
P(x_{t+1}=i \mid x_{1\ldots t})=\frac{\exp(v_i)}{\sum_j \exp(v_j)} \qquad (21)
$$

$$
v=\mathrm{TransformerLM}(x_{1\ldots t})_t \in \mathbb{R}^{\mathrm{vocab\_size}} \qquad (22)
$$

where TransformerLM is our model which takes as input a sequence of length `sequence_length` and produces a matrix of size `(sequence_length, vocab_size)`, 
and we take the last element of this matrix, as we are looking for the next token prediction at the $t$-th position.

This gives us a basic decoder by repeatedly sampling from these one-step conditionals (appending our previously-generated output token to the input of the next decoding timestep) 
until we generate the end-of-sequence token `<|endoftext|>` (or a user-specified maximum number of tokens to generate).

**Decoder tricks**

We will be experimenting with small models, and small models can sometimes generate very low-quality texts. 
Two simple decoder tricks can help fix these issues. 
First, in *temperature scaling* we modify our softmax with a temperature parameter $\tau$, where the new softmax is

$$
\mathrm{softmax}(v,\tau)_i=\frac{\exp(v_i/\tau)}{\sum_{j=1}^{\mathrm{vocab\_size}}\exp(v_j/\tau)}. \qquad (23)
$$

Note how setting $\tau \to 0$ makes it so that the largest element of $v$ dominates, and the output of the softmax becomes a one-hot vector concentrated at this maximal element.

Second, another trick is *nucleus* or *top-p* sampling [1], where we modify the sampling distribution by truncating low-probability tokens. 
Let $q$ be a probability distribution that we get from a (temperature-scaled) softmax of size `vocab_size`. 
Nucleus sampling with hyperparameter $p$ produces the next token according to the equation

$$
P(x_{t+1}=i \mid q)=
\begin{cases}
\frac{q_i}{\sum_{j\in V(p)}q_j} & \text{if } i\in V(p) \\
0 & \text{otherwise}
\end{cases}
\qquad (24)
$$

where $V(p)$ is the *smallest set* of indices such that $\sum_{j\in V(p)}q_j \geq p$. 
You can compute this quantity by sorting the probability distribution $q$ from largest to smallest, selecting tokens until the cumulative probability first reaches or exceeds $p$, zeroing out the remaining tokens, and renormalizing the kept probabilities before sampling. 
For example, if the sorted probabilities are `0.40, 0.25, 0.15, 0.10, ...` and `top_p = 0.80`, keep the first three tokens because `0.40 + 0.25 + 0.15 = 0.80`.

In code, generation should follow this order:

1. Run the model on the current context and take the logits for the last position.
2. If `temperature <= 0`, use greedy `argmax`.
3. Otherwise divide logits by `temperature`.
4. If `top_k` is set, keep only the `top_k` largest logits.
5. If `top_p` is set, apply the nucleus filter described above.
6. Apply `softmax` to the remaining logits and sample one token.
7. Append the sampled token and stop if it is `<|endoftext|>`.

Implement the generation functions in `release/minillm/model/generation.py`. 
Your implementation should support the following features:
- Generate completions for a user-provided prompt (i.e., take in some $x_{1\ldots t}$ and sample a completion until you hit an `<|endoftext|>` token).
- Allow the user to control the maximum number of generated tokens.
- Given a desired temperature value, apply softmax temperature scaling to the predicted next-token distributions before sampling.
- Top-$k$ sampling, where only the `k` highest-logit tokens remain available.
- Top-$p$ sampling, also referred to as nucleus sampling [1], given a user-specified threshold value.

## References

[1] A. Holtzman, J. Buys, L. Du, M. Forbes, and Y. Choi, “The Curious Case of Neural Text Degeneration,” in Proc. of ICLR, 2020.
