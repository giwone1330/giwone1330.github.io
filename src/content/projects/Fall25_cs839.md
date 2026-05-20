---
title: "Exploring Beyond VisRAG"
description: "We evaluated multimodal RAG on visually simple but text-heavy documents to analyze where VisRAG breaks down, how far text-only retrieval can go, and whether combining the two modalities produces a stronger retrieval system."
pubDate: "Dec 20 2025"
heroImage: "/projects/Fall25_cs839/front_page.png"
badge: "Pinned"
tags: ["RAG", "VLM", "Multimodal"]
# url: "/projects/Fall25_cs839/report.pdf"
---

*CS839 Foundation Models*

## Overview
Multimodal RAG is often evaluated on documents with charts, figures, and strong layout cues, where vision-heavy retrieval has an obvious advantage. In this project, we asked a harder and more practical question: what happens when the documents are visually simple but text-heavy? We studied that question by comparing pure visual retrieval, OCR-based text retrieval, and a hybrid image-text scoring method on document-image benchmarks.

The report's central claim held up clearly: on text-dominant documents, **visual retrieval alone is not competitive**, but visual signals can still improve a strong text retriever when they are used selectively rather than as the primary retrieval channel.

## Motivation

Vision-language retrieval is appealing because it preserves layout, typography, and page-level visual context. But that same design can become a weakness when relevance depends mostly on lexical and semantic matching. OCR-based text pipelines have the opposite tradeoff: they often retrieve better on text-heavy pages, but they can lose structural cues and break on noisy or stylized text.

We wanted to test that boundary directly and answer a more precise systems question: **should multimodal retrieval be expected to win by default, or only when the document actually contains useful visual evidence?**

## What we built

We built a lightweight evaluation pipeline around the archived experiment code rather than relying on the original large-scale VisRAG scripts.

For retrieval, we used the **NL-DIR** benchmark and compared three setups: pure visual retrieval with **VisRAG-Ret**, pure text retrieval using **DeepSeek-OCR** plus text embeddings such as **BGE 1.5** and **E5-small**, and a **hybrid fusion scorer** that combined image and text similarity with a tunable weight. The retrieval code uses OCR extraction, vector indexing with **LanceDB**, and evaluation with **MRR** and **Recall@10**.

For generation, we followed the VisRAG paper more loosely and built a smaller **single-GPU** pipeline using **MiniCPM-V 2.0** on a subsampled **MP-DocVQA** setup. That let us test a practical question that retrieval metrics alone cannot answer: whether better retrieval actually translates into better question answering once multiple retrieved pages are passed into the generator.

The fusion rule was intentionally simple: we combined normalized image and text similarity scores and swept the image weight from 0.0 to 1.0 in steps of 0.1 to see where visual information helped and where it started to hurt.

## Hybrid fusion implementation

This score-fusion method was the part of the project where my contribution was most direct. Instead of building a more complicated cross-modal reranker, I implemented a **late-fusion retriever** that queries the image and text indexes separately and then combines their scores after retrieval.

The core rule is:

$$
S_{\mathrm{hybrid}} = \alpha S_{\mathrm{image}} + (1 - \alpha) S_{\mathrm{text}}
$$

where $\alpha = 1.0$ gives image-only retrieval, $\alpha = 0.0$ gives text-only retrieval, and intermediate values interpolate between the two.

In implementation terms, the method works in five steps.

1. Query the **VisRAG image index** and the **OCR-text index** independently for the same natural-language query.
2. Retrieve raw LanceDB distance scores from both indexes.
3. Convert those distances into comparable per-query similarity scores using **min-max normalization**, with smaller distances mapped to higher normalized scores.
4. Form the union of document ids returned by either modality, assigning a score of `0.0` when a document is missing from one side.
5. Compute the weighted fused score for each candidate and sort in descending order.

The most important implementation detail is the normalization step. The image and text retrievers do not naturally operate on the same score scale, so simply averaging raw distances would be hard to justify. The code therefore converts each modality's output into a normalized `[0, 1]` similarity score before fusion:

$$
s_{\mathrm{norm}} = 1 - \frac{d - d_{\min}}{d_{\max} - d_{\min}}
$$

That makes the fusion interpretable and lets $\alpha$ act as a real weighting parameter instead of being distorted by incompatible score ranges.

The fusion loop itself is simple but important:

```python
image_scores = normalize_scores(image_results)
text_scores = normalize_scores(text_results)

for doc_id in set(image_scores) | set(text_scores):
	img_score = image_scores.get(doc_id, 0.0)
	txt_score = text_scores.get(doc_id, 0.0)
	fused_scores[doc_id] = alpha * img_score + (1 - alpha) * txt_score
```

I like this method because it is easy to reason about. If OCR works well, the text side dominates. If OCR fails on stylized or corrupted text, the visual side can rescue the document by contributing a nonzero image score. That is exactly the behavior we wanted from a hybrid retriever on text-heavy documents.

## Why the implementation mattered

One useful part of this project was that the team did not just run a benchmark from an existing repository. The archived code shows a more practical pipeline:

- OCR extraction with DeepSeek-OCR,
- separate image and text indexing,
- retrieval evaluation over saved datasets,
- and a smaller generation stage designed to run on a single L4 GPU rather than a distributed multi-GPU setup.

That mattered because it turned the project from a paper reproduction exercise into a systems diagnosis problem. We were not only asking which retriever scored higher, but also what broke when the pipeline had to run under realistic compute and OCR constraints.

## Key findings

### Retrieval on text-heavy documents

On text-heavy documents, pure text retrieval clearly outperformed the vision-only approach. Using BGE embeddings, TextRAG reached an **MRR of 0.7216** and **Recall@10 of 0.8310**, while VisRAG reached **0.1657 MRR** and **0.3010 Recall@10**. That gap showed that visual retrieval alone is not enough when the document's useful signal is overwhelmingly linguistic.

![Table1](/projects/Fall25_cs839/table1.png)

### Image-text fusion helped, but only when vision stayed weakly weighted

The most interesting result came from modality fusion. In the report's **500-query fusion sweep**, the pure-text baseline started at **0.8273 MRR** and **0.9200 Recall@10**. Adding a modest visual contribution improved over that baseline, with the best MRR appearing at **alpha = 0.30**, where the hybrid retriever reached **0.8379 MRR** and **0.9280 Recall@10**.

That is the most important systems result in the project, and it came directly from the fusion implementation above: the hybrid scorer surpassed both standalones in the same sweep. It beat the **text-only** setting at `alpha = 0.0` and the **image-only** setting at `alpha = 1.0`, which is exactly the result I was hoping to test. Visual signals were not strong enough to lead retrieval on their own, but they were still useful as a supporting signal, especially when OCR missed text or corrupted it. As the image weight increased further, MRR degraded steadily. Recall@10 peaked slightly later at **0.9300** when the weights were split **0.50 / 0.50**, but by then ranking quality had already started to fall.

![Table2](/projects/Fall25_cs839/table2.png)

### Better retrieval did not automatically produce better generation

We also found that stronger retrieval did not automatically lead to stronger downstream question answering. In the **30-query** generation setup, TextRAG performed best in the **Top-1** setting, reaching **12/30** accuracy, while VisRAG reached only **2/30**. But once more retrieved pages were concatenated, performance dropped for both systems. Even the oracle setting reached only **17/30**, which indicates the bottleneck had shifted from retrieval to the generation model's ability to reason over multi-page context.

This was a useful negative result. It showed that once retrieval improves, the next failure mode may be the generator rather than the retriever. In this case, the limiting factor appeared to be **MiniCPM-V 2.0**, not the retrieval stage alone.

![Table3](/projects/Fall25_cs839/table3.png)

### Visual retrieval still mattered in OCR failure cases

The report's error analysis explains why hybrid retrieval outperformed pure text retrieval despite the overall dominance of text embeddings. We found concrete OCR failure modes where visual retrieval preserved information that the OCR pipeline lost:

- missing text in stylized or decorative typography,
- mistranscriptions such as **"smoker" -> "speaker"**,
- and repetitive OCR loops that corrupted the text representation.

Those cases are exactly where the visual modality justified its place in the pipeline. The right conclusion was not that VisRAG should replace TextRAG, but that visual retrieval works best as an **error-correction signal** when OCR is brittle.

## My contribution

This was a team project, and I contributed across the broader system design and evaluation. My main contribution was designing and implementing the **hybrid score-fusion retrieval method**. Concretely, I built the retrieval routine that separately queries the image and OCR-text indexes, normalizes their scores, fuses them with a weighted late-fusion rule, and ranks the joint candidate set.

That implementation produced the most interesting result in the project: the fused retriever surpassed both the standalone OCR-based text setup and the standalone VisRAG setup when the visual signal was given a modest weight. In other words, my contribution was not just proposing the idea of fusion in the abstract, but showing through implementation and evaluation that a simple weighted scorer could outperform either modality alone.

I also spent a lot of time interpreting the failure cases rather than treating the metric tables as the end of the story, because the OCR breakdowns were what actually explained why the hybrid setup worked. The result only makes sense once you see the pattern: text retrieval is usually stronger on these documents, but visual retrieval becomes valuable exactly when OCR makes the kinds of mistakes that damage semantic matching.

## Conclusion

This project strengthened my understanding of multimodal retrieval at a systems level: not just how to build a RAG pipeline, but how to diagnose when a modality is helping, when it is hurting, and how to combine signals in a principled way.

The practical lesson I took from it is one I keep seeing across ML systems work: the best architecture is often not the most complex one, but the one that matches the structure of the data. On visually simple, text-heavy documents, text retrieval should do most of the work. Vision still matters, but only when it is used in a targeted way that compensates for OCR failures rather than trying to dominate the full retrieval problem.

