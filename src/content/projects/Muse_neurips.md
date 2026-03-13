---
title: "Multilinear Operators for Arithmetic Tasks"
description: "This project explored whether sequence-to-sequence models could be built entirely from multilinear operations, without softmax attention, nonlinear activations, or positional encodings, and resulted in MuSe, a multilinear sequence model designed for arithmetic tasks such as addition, multiplication, and parity."
pubDate: "Aug 31 2025"
heroImage: "/projects/Muse_neurips/figure1.png"
badge: "Pinned"
tags: ["Polynomial networks", "Tensor operators"]
url: "https://drive.google.com/file/d/1ubnWU1zOyUdkixESXEFawG-icBS_oAmS/view?usp=drive_link"
---


*Lead research project · Jan. 2025 – Aug. 2025 · NeurIPS 2025 submission*

This project explored whether sequence-to-sequence models could be built entirely from multilinear operations, avoiding softmax attention, nonlinear activations, and positional encodings. I led the development of **MuSe**, a multilinear sequence model designed for arithmetic tasks such as addition, multiplication, and parity. At the time, the project looked like an alternative architecture paper. In retrospect, its deeper value was that it pushed me toward a more serious question: **what kinds of sequence models remain viable when the computation itself must stay algebraically simple?**

## Motivation

The original technical motivation came from privacy-preserving machine learning. Standard Transformers rely heavily on softmax attention and nonlinear activations, both of which are awkward for **fully homomorphic encryption** settings where addition and multiplication are the natural operations. MuSe was an attempt to move toward a sequence model that is more compatible with that world.

In the paper, arithmetic tasks served as a controlled testbed rather than the final application. They let me study whether a multilinear architecture could learn causal sequence structure, generalize to longer inputs, and remain competitive with Transformer baselines even without positional encodings. Looking back, this core motivation should have been stated much more clearly and much earlier.

## What I built

![figure1](/projects/Muse_neurips/figure1.png)

I led the project and drove the main research direction. The core idea was a **polynomial attention** mechanism that captures causal token interactions through tensor operations instead of softmax. I combined that attention module with **Mu-layers** in a decoder-style architecture, forming MuSe as a fully multilinear sequence model.

![polyattn](/projects/Muse_neurips/polyattn.png)

A key design choice was to intentionally avoid both nonlinear activations and positional encodings. That made the model more restrictive than a standard Transformer, but it also made the architecture much closer to the kinds of operations that matter for encrypted inference. I evaluated the model against decoder-only Transformer baselines, including variants with and without positional encodings, on synthetic arithmetic tasks covering **addition, multiplication, and parity**.

![table12](/projects/Muse_neurips/table12.png)

## Key findings

MuSe produced encouraging results on the targeted benchmark. In the paper, it showed strong performance on arithmetic reasoning despite operating without positional encodings. The model generalized from training on **30-digit addition** to evaluation on **90-digit addition**, and it also showed faster learning behavior than the Transformer baselines on several arithmetic settings. On multiplication and parity tasks, MuSe remained competitive while using a much more constrained class of operations.

![figure2](/projects/Muse_neurips/figure2.png)
![figure3](/projects/Muse_neurips/figure3.png)

Those results convinced me that the architecture was not just mathematically interesting, but experimentally promising. Even though the paper did not get accepted, I still see the technical outcome as meaningful: it showed that causal sequence modeling does not have to begin and end with standard Transformer design choices.

## My role

This was primarily **my** project. I led the research direction, owned the overall technical story, and was responsible for pushing the work toward a publishable paper. Other collaborators helped with some of the experiments and parts of the writing, but this was the first project where I had to take responsibility not only for implementation, but also for the motivation, framing, and scientific argument.

## What the rejection taught me

This paper was submitted to **NeurIPS 2025** and was rejected. In hindsight, that rejection was useful because it exposed weaknesses that the experimental curves alone could not hide.

### Motivation has to be explicit, not implied

The most important review question was simple: **why should anyone use this model?** I had not answered that well enough. The paper mentioned privacy-sensitive settings and homomorphic encryption, but the motivation was not developed into a strong central argument. The result was a technically novel model whose broader purpose still felt underexplained.

### Design decisions need intuition, not just equations

I could describe the architecture mathematically, but I had not done enough to explain **why this specific design** made sense. Reviewers wanted intuition for the architectural choices, not just formulas and implementation details. That was a fair criticism.

### Terminology has to be exact

One review correctly pointed out that I used **sample complexity** incorrectly. What I was really describing was faster convergence under a fixed training setup, not sample complexity in the formal sense. That was my mistake, and it taught me how much precision in language matters for credibility.

### Positioning against related work matters

Another fair criticism was that I had not positioned the work clearly enough against nearby ideas in efficient sequence modeling, including gated-convolution-style approaches. Even when a design is meaningfully different, that difference has to be argued proactively. I learned that novelty is not something reviewers are supposed to infer on my behalf.

### Scope alone does not settle robustness concerns

One reviewer asked about tasks beyond arithmetic, such as language modeling. I still think that critique partly missed the fact that the paper explicitly scoped itself to arithmetic tasks. At the same time, it forced me to confront a broader issue: if the intended motivation is serious, then the benchmark has to feel clearly connected to that motivation. A narrow scope is acceptable, but only if the reader understands why that scope is the right first step.

## How this changed my research direction

This project changed how I think about research ownership. The initial project direction came from my advisor, and at first I treated it as something to complete well. The rejection made it obvious that executing a project is not the same as truly owning it. To lead research well, I have to understand not only how the model works, but also why it exists, what assumptions it makes, where it fits in the literature, and how to defend every major claim.

That lesson directly shaped my next step. I took the multilinear modeling direction and connected it much more explicitly to **homomorphic encrypted inference**, which is now the foundation of my master’s thesis: **"Accelerating Homomorphic Encrypted Inference with Multilinear Networks."** Instead of treating privacy compatibility as a side note, I now treat it as the core problem setting.

## Why this project still matters

Even though the submission was rejected, I consider this one of the most important projects in my development as a researcher. It was my first experience leading a research effort, building a new sequence architecture, learning how fragile scientific framing can be, and understanding what it means to become a self-motivated independent researcher.

In that sense, the paper did not fail. It did what good research projects sometimes do: it showed me both the promise of an idea and the limits of my own understanding, then forced me to grow into the next version of the work.
