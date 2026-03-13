---
title: "Inducing and Analyzing Hallucination in Diffusion LLMs"
description: "We studied how factual hallucinations can be induced in diffusion-based language models—which iteratively denoise entire sequences rather than generate tokens autoregressively—to understand what these failures reveal about their internal behavior."
pubDate: "Dec 20 2025"
heroImage: "/projects/Fall25_cs762/CS762_Poster_print.jpg"
badge: "Pinned"
tags: ["Hallucination", "Diffusion LLM"]
# url: "https://github.com"
---



*CS762 Advanced Deep Learning*

## Overview
Diffusion-based language models offer a fundamentally different generation process from standard autoregressive LLMs. Instead of predicting one token at a time, they iteratively denoise an entire sequence in parallel. That makes them interesting not only for generation efficiency, but also for reliability research. In this project, we studied how factual hallucinations can be induced in a diffusion LLM and what those failures reveal about the model’s internal behavior.

## Motivation

Most hallucination research focuses on autoregressive models, but diffusion LLMs introduce a different decoding dynamic. We wanted to understand whether small token-level interventions could systematically bias these models toward incorrect answers, and whether those errors would stay local or propagate across the full response.

## What we built

We designed a controlled evaluation pipeline around **LLaDA-8B-Instruct** and curated a benchmark of **500 factual questions** spanning five categories: country capitals, birth years, birth places, event locations, and food origins. Rather than asking the model to denoise a fully masked sequence, we inserted a single **anchor token** into the response and measured how that intervention changed the final answer.

We tested two main settings. In the first, we placed an incorrect anchor at the expected answer location to see whether the model would resist the false signal or commit to it. In the second, we placed anchors at different positions throughout the sequence to measure how position alone affected hallucination behavior.

## Key findings

![Figure 1](/projects/Fall25_cs762/figure1.png)

The results were unexpectedly strong. When an incorrect token was inserted at the key answer position, hallucination became almost unavoidable in most categories. Birth place and food origin questions produced direct hallucinations every time, while birth year questions reached a 97% direct hallucination rate.

![Table 1](/projects/Fall25_cs762/table1.png)


Even more interesting, the effect was not limited to answer-critical positions. Across early, middle, and late anchor placements, hallucination rates stayed above 80%. That suggests the injected token does not just distort a local span of text. In a diffusion model, it can reshape the global denoising trajectory and steer the entire response toward false or fabricated content.

![Figure 2](/projects/Fall25_cs762/figure2.png)

We also found a qualitative difference across tasks. Country-capital questions often triggered **indirect hallucinations** rather than direct ones. Instead of stating a blatantly false answer, the model frequently preserved a locally true statement while silently shifting the subject of the response. That behavior points to a distinctive failure mode in instruction-tuned diffusion models: they may preserve learned answer templates even when the conditioning conflicts with the original question.


<div class="grid grid-cols-2 gap-4">
  <img src="/projects/Fall25_cs762/core_example.png" alt="Image 1">
  <img src="/projects/Fall25_cs762/partial_example.png" alt="Image 2">
</div>





## My contribution

This was a team project, and I contributed across the overall effort. My main focus was on **running experiments** and **analyzing failure modes across topics and anchor positions**. In practice, that meant comparing how hallucinations varied by question category, studying how early, middle, and late anchor placements changed model behavior, and helping interpret the difference between direct and indirect hallucinations.

## Internal model behavior

To probe whether the model still retained the correct answer internally, we analyzed token-level logit distributions during denoising. For birth-year questions, the correct answer often remained the second- or third-ranked option even when the final output was wrong. For birth-place questions, the correct city received very low probability throughout the process, suggesting a more confident and deeply committed hallucination.

This gap between internal knowledge and final decoded output was one of the most interesting takeaways from the project. It suggests that some hallucinations in diffusion LLMs are not simply failures of stored knowledge, but failures of how that knowledge is expressed under constrained generation.

## Conclusion

This work gave me hands-on experience with LLM reliability evaluation, experimental analysis, and decoding-time failure analysis. More importantly, it reinforced a research direction I care deeply about: building language systems that are not only capable, but also interpretable and trustworthy under real-world constraints.

As diffusion-based language models continue to mature, understanding their unique failure modes will be essential. This project was an important step in that direction.
