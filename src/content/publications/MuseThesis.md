---
title: "MuSe: Packing-Aware Causal Sequence Modeling under Fully Homomorphic Encryption"
description: "This master's thesis turns MuSe from an alternative multilinear sequence model into a packing-aware architecture for pure-FHE causal inference under bootstrapped CKKS, with both outer- and inner-based packing and cache-enabled generation."
pubDate: "May 18 2026"
heroImage: "/publications/musethesis/figure1-overview.png"
badge: "Pinned"
tags: ["Thesis", "Homomorphic Encryption", "Sequence Modeling", "Tensor operators"]
venue: "Master's Thesis · University of Wisconsin–Madison"
doi: "https://drive.google.com/file/d/1eSPaASRvVPgcqKmsVjx5fd-ehwWu81t2/view?usp=drive_link"
url: "https://github.com/giwone1330/MuSe"
---

## Overview

This thesis takes the earlier MuSe project into a much more concrete direction. Instead of asking only whether a multilinear sequence model can work on controlled tasks, the thesis asks whether that same modeling idea can be turned into a **packing-aware architecture for fully homomorphic causal inference**. The central problem is not just model design in isolation. It is the interaction between operator choice, ciphertext layout, causal generation, and runtime cost under bootstrapped CKKS.

The result is a new framing for MuSe: a decoder-style sequence model whose core computations are designed to stay closer to the arithmetic that encrypted inference can support. In the thesis, MuSe replaces softmax attention with a softmax-free causal mixer and replaces the standard feed-forward block with a multilinear operator, then studies how that architecture behaves under both **outer-based packing** and **inner-based packing** inside the desilo FHE stack.


<div class="grid grid-cols-1 gap-4">
	<figure class="m-0">
		<img src="/publications/musethesis/figure1-overview.png" alt="Cropped thesis figure showing the secure client-server inference pipeline for MuSe, from plaintext input and encryption on the client to encrypted inference on the server and decryption on the client." />
		<figcaption class="mt-2 text-center text-xs leading-snug opacity-80">Figure 1. Secure client-server MuSe inference pipeline from the thesis</figcaption>
	</figure>
</div>

Three outcomes mattered most in the final thesis:

1. MuSe supports end-to-end pure-FHE causal inference under both outer-based and inner-based packing on the reported 3-digit addition setting.
2. Inner-based cached generation reduces per-token runtime from 5,639 seconds to 468 seconds, a 12.05x speedup, while keeping the cached-generation cosine distance below $10^{-8}$.
3. In plaintext evaluation, matched MuSe variants use 56.3% to 76.8% fewer non-embedding parameters than the Transformer references while remaining competitive on selected downstream evaluations.

## Why This Thesis Was Different

The earlier MuSe project was primarily about whether a multilinear sequence architecture could learn useful causal structure at all. This thesis is where that idea became a systems question. The main issue is no longer only model novelty, but whether the architecture is actually better aligned with the constraints of fully homomorphic encrypted inference.

That shift changed the project in two important ways. First, the architecture was evaluated through the lens of **packing-aware encrypted execution**, not just plaintext benchmarks. Second, the thesis separated two different claims that are easy to blur together: whether MuSe is a compact and useful plaintext sequence model, and whether MuSe is a more workable architecture for pure-FHE causal generation than a standard softmax-based decoder.

## Core Architecture

At the architectural level, the thesis still keeps a Transformer-like decoder structure, but replaces the two most important sublayers. Standard self-attention is replaced by the Polynomial Toeplitz Mixer, and the usual feed-forward network is replaced by a multilinear operator. That matters because the design is no longer only about model novelty. It is about whether the main sequence-mixing and feature-mixing steps can be expressed in a form that is more natural for encrypted arithmetic.

<div class="grid grid-cols-1 gap-4">
	<figure class="m-0">
		<img src="/publications/musethesis/figure2-ptm.png" alt="Cropped thesis figure illustrating the Polynomial Toeplitz Mixer and how its tensors are constructed and contracted." />
		<figcaption class="mt-2 text-center text-xs leading-snug opacity-80">Figure 2. Polynomial Toeplitz Mixer overview from the thesis</figcaption>
	</figure>
</div>

The Polynomial Toeplitz Mixer is the core sequence operator in the thesis. Instead of building attention around softmax-normalized score matrices, it constructs a causal lower-triangular mixing pattern directly through multilinear tensor operations. That is the operator-level shift that lets the thesis talk about MuSe not just as an alternative sequence model, but as a candidate architecture for pure-FHE causal inference.

## Plaintext Model Evaluation

Even though the thesis is centered on encrypted inference, it still had to show that MuSe was not just a cryptographic artifact. The plaintext study therefore split the evidence into pretrained scaling, zero-shot benchmark behavior, fine-tuned downstream evaluation, and one supplementary representation analysis.

<div class="grid grid-cols-1 gap-4">
	<figure class="m-0">
		<img src="/publications/musethesis/table3-model-suite.png" alt="Cropped Table 3 from the thesis showing the MuSe and Transformer model suites and their non-embedding parameter counts." />
	</figure>
</div>

This table is the basis for most of the plaintext comparisons in the thesis. The important point is not just that MuSe uses fewer parameters, but how much fewer: depending on model size, the matched MuSe variants use 56.3% to 76.8% fewer non-embedding parameters than the Transformer references.

<div class="grid grid-cols-1 gap-4">
	<figure class="m-0">
		<img src="/publications/musethesis/table4-validation-loss.png" alt="Cropped Table 4 from the thesis showing pretrained validation loss across MuSe and Transformer model sizes." />
	</figure>
</div>

<div class="grid grid-cols-1 gap-4">
	<figure class="m-0">
		<img src="/publications/musethesis/scaling-perplexity.png" alt="Validation perplexity versus parameter count for pretrained MuSe and Transformer checkpoints across four model sizes." />
		<figcaption class="mt-2 text-center text-xs leading-snug opacity-80">Validation perplexity across the pretrained scaling sweep</figcaption>
	</figure>
</div>

On raw validation perplexity, the Transformer references still performed better. That matters, and the thesis does not hide it. But the scaling results also show that MuSe remains stable as model size increases rather than collapsing as capacity grows. The more important thesis claim is narrower: MuSe can still occupy useful **performance-to-parameter tradeoffs** even when it is not the strongest model by raw language-modeling loss.

<div class="grid grid-cols-1 gap-4">
	<figure class="m-0">
		<img src="/publications/musethesis/table5-zero-shot-results.png" alt="Cropped Table 5 from the thesis showing zero-shot LM evaluation results for pretrained FineWeb checkpoints." />
	</figure>
</div>

That tradeoff shows up more clearly in the zero-shot and fine-tuned results than in perplexity alone. Table 5 is the important evidence here because it shows the full benchmark sweep rather than only the most favorable plots. In the thesis, MuSe remained favorable on a subset of zero-shot tasks such as the ANLI rounds, ARC Easy, BoolQ, and TruthfulQA MC2, while several other tasks stayed effectively tied.

<div class="grid grid-cols-1 gap-4">
	<figure class="m-0">
		<img src="/publications/musethesis/table6-arithmetic-results.png" alt="Cropped Table 6 from the thesis showing fine-tuned arithmetic accuracy for the smallest shared MuSe and Transformer configurations." />
	</figure>
</div>

<div class="grid grid-cols-1 gap-4">
	<figure class="m-0">
		<img src="/publications/musethesis/table7-nlp-results.png" alt="Cropped Table 7 from the thesis showing fine-tuned NLP accuracy for the smallest shared MuSe and Transformer configurations." />
	</figure>
</div>

Tables 6 and 7 show the clearest plaintext argument in the thesis. MuSe was especially strong on the structured arithmetic tasks, but it also stayed competitive on selected NLP settings, with much larger margins on SNLI and IMDb than on AG News or BoolQ. That combination is why the thesis can describe MuSe as a compact sequence model rather than only as an encrypted-systems experiment.

<div class="grid grid-cols-1 gap-4">
	<figure class="m-0">
		<img src="/publications/musethesis/imdb-perturbation.png" alt="Perturbation-based representation analysis comparing hidden-state sensitivity for MuSe and Transformer on IMDb." />
		<figcaption class="mt-2 text-center text-xs leading-snug opacity-80">Supplementary representation analysis on IMDb</figcaption>
	</figure>
</div>

The thesis also included a perturbation-based representation analysis on IMDb as supplementary architectural evidence. Under importance-ranked masking, MuSe's final hidden state moved much faster than the matched Transformer baseline. I do not read that as a universal interpretability claim. What it did show is that the multilinear architecture organizes token information differently, and that difference remains visible even outside the encrypted setting.

## Packing-Aware Encrypted Evaluation

The core thesis contribution is the encrypted systems study. Here the question is whether MuSe can support **pure-FHE causal generation** under more than one ciphertext layout, and what runtime, level-consumption, and fidelity tradeoffs appear once packing becomes a first-class design constraint.

The thesis studies two packing regimes. In **outer-based packing**, more sequence positions are handled in parallel, which is attractive for fixed-length evaluation but awkward for dynamic causal generation. In **inner-based packing**, one causal step is mapped more directly into the ciphertext layout, which is less SIMD-friendly but much better aligned with cache-style generation. That distinction is one of the main reasons this thesis feels different from the earlier MuSe project: packing is treated as part of the architectural question, not as a low-level implementation detail.

<div class="grid grid-cols-1 gap-4">
	<figure class="m-0">
		<img src="/publications/musethesis/table9-cosine-distance.png" alt="Cropped Table 9 from the thesis showing hidden-state cosine-distance summaries across packing methods." />
	</figure>
</div>

<div class="grid grid-cols-1 md:grid-cols-2 gap-4">
	<figure class="m-0">
		<img src="/publications/musethesis/fhe-block-cosine.png" alt="Blockwise cosine-distance comparison between encrypted and plaintext MuSe hidden states under different packing layouts." />
		<figcaption class="mt-2 text-center text-xs leading-snug opacity-80">Blockwise encrypted versus plaintext fidelity</figcaption>
	</figure>
	<figure class="m-0">
		<img src="/publications/musethesis/fhe-component-cosine.png" alt="Componentwise cosine-distance comparison between encrypted and plaintext MuSe execution under different packing layouts." />
		<figcaption class="mt-2 text-center text-xs leading-snug opacity-80">Componentwise encrypted versus plaintext fidelity</figcaption>
	</figure>
</div>

One of the most important results is that encrypted execution stayed close to the plaintext reference under all three reported realizations. Table 9 makes that visible immediately. The inner-based configurations remained especially close, with last-block hidden-state cosine distances of 0.000205 for fused inner packing and 0.000182 for unfused inner packing. Outer-based packing drifted more, but still remained controlled, ending at 0.022635 in the final block. That is enough to make the packing discussion concrete: the layouts are not interchangeable, but neither one causes the model to numerically fall apart.

<div class="grid grid-cols-1 gap-4">
	<figure class="m-0">
		<img src="/publications/musethesis/fhe-level-per-ct.png" alt="Per-ciphertext level consumption across outer-based and inner-based packing layouts for MuSe blocks under encrypted inference." />
		<figcaption class="mt-2 text-center text-xs leading-snug opacity-80">Per-ciphertext level consumption across packing layouts</figcaption>
	</figure>
</div>

The level-consumption plot shows why the packing decision matters operationally. Outer-based packing consumed far more levels per ciphertext than either inner-based variant, largely because normalization required more iterations to converge in that layout. The fused inner-based version reduced that burden further by moving eligible linear work out of the encrypted online path. In other words, the operator design helped, but the actual encrypted operating point still depended heavily on how the model was packed.

<div class="grid grid-cols-1 md:grid-cols-2 gap-4">
	<figure class="m-0">
		<img src="/publications/musethesis/fhe-cache-cosine.png" alt="Cosine distance between cached and non-cached encrypted generation for inner-based MuSe inference." />
		<figcaption class="mt-2 text-center text-xs leading-snug opacity-80">Cached versus non-cached hidden-state fidelity</figcaption>
	</figure>
	<figure class="m-0">
		<img src="/publications/musethesis/fhe-cache-runtime.png" alt="Runtime breakdown for cached inner-based MuSe generation under encrypted inference." />
		<figcaption class="mt-2 text-center text-xs leading-snug opacity-80">Cached generation runtime breakdown</figcaption>
	</figure>
</div>

The strongest systems result in the thesis came from cache-enabled generation under inner-based packing. Once previously computed encrypted context could be reused instead of recomputing the full prefix each step, per-token generation time dropped from 5,639 seconds to 468 seconds. That 12.05x speedup is the clearest evidence in the thesis that MuSe is not just softmax-free in principle, but can actually support a more workable causal-generation pattern under pure FHE.

## What I Learned

This thesis made one lesson much clearer than the earlier MuSe paper did: a model idea becomes much stronger once it is tied to a real systems constraint. In the earlier project, multilinear sequence modeling was interesting, but the motivation still felt partially abstract. In the thesis, the same architecture became more defensible because it was evaluated against a concrete question about bootstrapped CKKS, packing layouts, and cache-enabled encrypted generation.

It also changed how I think about architectural efficiency. A model can be weaker than a Transformer on raw pretraining loss and still be meaningful if it occupies a better operating point for the problem that actually matters. For this thesis, that meant looking at plaintext performance relative to parameter count, then connecting that compactness to encrypted feasibility rather than pretending those are unrelated stories.

Finally, the thesis made the systems side of sequence modeling feel much more real. Packing is not just a storage choice. It changes runtime, level consumption, numerical drift, and whether cache reuse is practical at all. That is probably the main reason this project matters to me: it is where an earlier research direction turned into a concrete machine learning systems problem that I could analyze end to end.