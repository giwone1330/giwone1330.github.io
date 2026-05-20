---
title: "Inducing and Analyzing Hallucination in Diffusion LLMs"
description: "We studied how factual hallucinations can be induced in diffusion-based language models—which iteratively denoise entire sequences rather than generate tokens autoregressively—to understand what these failures reveal about their internal behavior."
pubDate: "Dec 20 2025"
heroImage: "/projects/Fall25_cs762/CS762_Poster_print.jpg"
badge: "Pinned"
tags: ["Hallucination", "Diffusion LLM"]
# url: "/projects/Fall25_cs762/report.pdf"
---



*CS762 Advanced Deep Learning*

## Overview
Diffusion-based language models offer a fundamentally different generation process from standard autoregressive LLMs. Instead of predicting one token at a time, they iteratively denoise an entire sequence in parallel. That makes them interesting not only for generation efficiency, but also for reliability research.

In this project, we studied whether a diffusion LLM could be **steered into factual hallucination by a single injected token**, and whether the model still retained the correct answer internally even when the final decoded output was wrong. The final report frames this as both a robustness study and an interpretability study: not just whether hallucination happens, but how the denoising process responds when local token constraints conflict with factual knowledge.

## Motivation

Most hallucination research focuses on autoregressive models, but diffusion LLMs introduce a different decoding dynamic. We wanted to understand whether small token-level interventions could systematically bias these models toward incorrect answers, and whether those errors would stay local or propagate across the full response.

## Experimental design

We designed a controlled evaluation pipeline around **LLaDA-8B-Instruct** and curated a benchmark of **500 factual questions** spanning five categories: country capitals, birth years, birth places, event locations, and food origins. Rather than asking the model to denoise a fully masked sequence, we inserted exactly one **anchor token** into the response and measured how that intervention changed the final answer.

We tested three closely related settings.

1. **Key-position anchors**: place an incorrect token at the location where the factual answer is normally expected.
2. **Position-shifted adversarial anchors**: place the injected token at positions `0, 1, 2, 3, 7, 10, 15, 30, 60` to test whether the effect stays local or propagates through the denoising process.
3. **Normal-anchor controls**: repeat the positional experiment with non-adversarial anchor tokens to check whether the failure is caused by anchoring in general or by semantically conflicting anchor content.

For the key-position setting, the report classifies outputs as **direct hallucination** or **indirect hallucination**. For the positional experiments, the analysis becomes finer-grained and distinguishes **core hallucination**, **partial hallucination**, **core and partial hallucination**, **parse error**, and **no hallucination**.

## Key findings

### Wrong anchors at answer positions are highly effective

![Figure 1](/projects/Fall25_cs762/figure1.png)

The results were unexpectedly strong. When an incorrect token was inserted at the key answer position, hallucination became almost unavoidable in most categories. **Person Birth Place** and **Food Origin** produced **100% direct hallucination**, while **Person Birth Year** reached **97% direct hallucination**. The main exception was **Country Capital**, where the model often produced **indirect hallucinations** instead of directly stating the false answer.

![Table 1](/projects/Fall25_cs762/table1.png)

That distinction mattered. Country-capital questions often did not collapse into a plainly false sentence like "The capital of France is Tokyo." Instead, the model frequently preserved a locally true statement while silently shifting the subject of the answer. The report treats this as a different failure mode from direct factual corruption.

### Adversarial anchors remain potent even away from the answer slot

Even more interesting, the effect was not limited to answer-critical positions. When adversarial anchors were moved across the output sequence, the report still found **hallucination rates above 80% regardless of anchor position**. That suggests the injected token does not just distort a local span of text. In a diffusion model, it can reshape the global denoising trajectory and steer the entire response toward false or fabricated content.

![Figure 2](/projects/Fall25_cs762/figure2.png)

### The control experiment weakens that effect

The normal-anchor control gave a more nuanced picture. When we used non-adversarial anchors at different positions, hallucination dropped substantially for most datasets. The major exception was **Event Location**, which remained relatively fragile. This control matters because it shows the failure is not simply caused by fixing any token in the sequence. The stronger effect comes from **semantically conflicting anchors** that push the denoising process toward a false completion.

![Figure 3](/projects/Fall25_cs762/figure3.png)

Taken together, Figure 2 and Figure 3 support a sharper claim than the original draft: diffusion models are vulnerable not merely to anchoring, but specifically to **adversarial anchoring** that conflicts with the underlying fact pattern.


<div class="grid grid-cols-2 gap-4">
  <img src="/projects/Fall25_cs762/core_example.png" alt="Image 1">
  <img src="/projects/Fall25_cs762/partial_example.png" alt="Image 2">
</div>





## My contribution

This was a team project, and I contributed across the overall effort. My main focus was on **running the experiments** and **analyzing failure modes across topics and anchor configurations**. In practice, that meant comparing how hallucinations varied by question category, studying how key-position anchors differed from position-shifted anchors, and helping interpret the difference between direct, indirect, core, and partial hallucinations.

I also contributed to the analysis of the control setting with normal anchors and to the token-level probing work. That part was important because it moved the project beyond simply showing that hallucinations can be induced. It helped us separate three different questions: whether the anchor position matters, whether the anchor semantics matter, and whether the correct answer remains recoverable inside the model even when the final generation is wrong.

## Internal model behavior

To probe whether the model still retained the correct answer internally, we analyzed token-level logit distributions during denoising. For birth-year questions, the correct answer often remained the second- or third-ranked option even when the final output was wrong. For birth-place questions, the correct city received very low probability throughout the process, suggesting a more confident and deeply committed hallucination.

This gap between internal knowledge and final decoded output was one of the most interesting takeaways from the project. It suggests that some hallucinations in diffusion LLMs are not simply failures of stored knowledge, but failures of how that knowledge is expressed under constrained generation.

## Conclusion

This work gave me hands-on experience with LLM reliability evaluation, experimental analysis, and decoding-time failure analysis. More importantly, it reinforced a research direction I care deeply about: building language systems that are not only capable, but also interpretable and trustworthy under real-world constraints.

As diffusion-based language models continue to mature, understanding their unique failure modes will be essential. The project’s strongest lesson for me was that a single conflicting token can have surprisingly global consequences in a parallel denoising model, but also that those failures are not uniform. Some categories remain internally recoverable, while others collapse into confident hallucination. That kind of distinction is exactly what future reliability work needs to explain.
