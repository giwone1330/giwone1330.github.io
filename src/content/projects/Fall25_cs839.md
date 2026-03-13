---
title: "Exploring Beyond VisRAG"
description: "We evaluated multimodal RAG on visually simple but text-heavy documents to analyze where VisRAG breaks down, how far text-only retrieval can go, and whether combining the two modalities produces a stronger retrieval system."
pubDate: "Dec 20 2025"
heroImage: "/projects/Fall25_cs839/front_page.png"
badge: "Pinned"
tags: ["RAG", "VLM", "Multimodal"]
# url: "https://github.com"
---

*CS839 Foundation Models*

## Overview
Multimodal RAG is often evaluated on documents with charts, figures, and layout cues, where vision-heavy retrieval has a clear advantage. In this project, we asked a more practical question: what happens when the documents are visually simple but text-heavy? We evaluated where VisRAG breaks down, how far text-based retrieval can go, and whether the two modalities can be fused into a stronger retrieval system.

## Motivation

Vision-language retrieval is appealing because it preserves page structure and visual context, but that same design can become a weakness when relevance depends mostly on lexical and semantic matching. We wanted to test that boundary directly on text-dominant document images and understand whether multimodal retrieval should be competitive by default, or only under specific document conditions.

## What we built

We compared three retrieval setups on document-image benchmarks: pure visual retrieval with **VisRAG-Ret**, pure text retrieval using **DeepSeek-OCR** plus text embeddings such as **BGE 1.5** and **E5-small**, and a **hybrid fusion scorer** that combined image and text similarity with a tunable weight. We also ran downstream generation experiments with **MiniCPM-V 2.0** on a subsampled **MP-DocVQA** pipeline to study how retrieval quality translated into question answering.

## Key findings

On text-heavy documents, pure text retrieval clearly outperformed the vision-only approach. Using BGE embeddings, TextRAG reached an **MRR of 0.7216** and **Recall@10 of 0.8310**, while VisRAG reached **0.1657 MRR** and **0.3010 Recall@10**. That gap showed that visual retrieval alone is not enough when the document’s useful signal is overwhelmingly linguistic.

![Table1](/projects/Fall25_cs839/table1.png)

The most interesting result came from modality fusion. By combining normalized image and text similarity scores, we found that a modest visual contribution improved over the text-only baseline. The best setting appeared at **α = 0.30**, where the hybrid retriever reached **0.8379 MRR** and **0.9280 Recall@10**. In other words, visual signals were not strong enough to lead retrieval on their own, but they were still valuable as a supporting signal, especially when OCR made mistakes or missed stylized text.

![Table2](/projects/Fall25_cs839/table2.png)

We also found that stronger retrieval did not automatically lead to stronger generation. In the downstream QA setup, TextRAG performed best in the Top-1 setting, but adding more retrieved pages often reduced accuracy. That suggested the bottleneck had shifted from retrieval to the generation model’s ability to reason over concatenated multi-page context.

![Table3](/projects/Fall25_cs839/table3.png)

## My contribution

This was a team project, and I contributed across the broader system design and evaluation. My main contribution was proposing and experimenting with the **modality-fusion scoring** approach. I focused on testing how different fusion weights changed retrieval quality and on interpreting what those trends said about the strengths and weaknesses of each modality. That work helped show that **VisRAG alone is a poor fit for text-heavy documents**, but that its visual signal can still strengthen a text-based RAG pipeline when used selectively through fusion.

## Conclusion

This project strengthened my understanding of multimodal retrieval at a systems level: not just how to build a RAG pipeline, but how to diagnose when a modality is helping, when it is hurting, and how to combine signals in a principled way. It also reinforced a practical lesson I keep returning to in ML systems work: the best architecture is often not the most complex one, but the one that matches the structure of the data.

