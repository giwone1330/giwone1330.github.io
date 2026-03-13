---
title: "Gaze-aware Proactive AI Assistance for Digital Article Readers"
description: "This team project developed a gaze-aware reading assistant for digital articles that uses webcam-based gaze tracking to detect when a reader is likely stuck and proactively provide AI support such as definitions, simplifications, and summaries within a practical browser setting."
pubDate: "May 20 2025"
heroImage: "/post_img.webp"
# heroImage: "/projects/Spring25_cs770/a.jpg"
# badge: "Pinned"
tags: ["HCI", "Gaze Detection"]
# url: "https://github.com"
---

*CS 770 Human Computer Interaction*

## Overview
This was a team project on building a **gaze-aware reading assistant** for digital articles. We explored whether webcam-based gaze tracking could be used to detect when a reader is likely stuck and then trigger **proactive AI support** such as definitions, simplifications, and summaries. The project sat at the intersection of HCI, eye tracking, and LLM-based assistance, with a strong emphasis on making the system work in a practical browser setting rather than relying on specialized hardware.

## Motivation

Reading dense digital text can be cognitively demanding, especially for users reading in a second language. Traditional reading aids are usually reactive: the user has to stop, highlight something, and explicitly ask for help. We wanted to study whether a system could use gaze behavior as an implicit signal of difficulty and provide support at the right time without breaking the reading flow.

## What we built

We designed a **Chrome extension** that combined three main components.

### Real-time gaze tracking

The first component estimated gaze from a standard webcam using a pipeline built around **face and eye landmark detection, gaze estimation, and document-level localization**. The system used webcam input, eye-region features, and calibration to map gaze to on-screen coordinates. It then generated gaze heatmaps and linked those coordinates back to the document so the system could infer what paragraph or region the reader was attending to.

### Assistance evaluation

The second component decided when help might be needed. We used gaze signals such as **long fixations, regressions, changes in reading speed, and scanning patterns** as indicators that the user might be struggling. Those signals were combined with contextual factors such as text difficulty to determine when to trigger support.

### Proactive AI assistance

The third component generated the actual assistance. Once the system identified the relevant reading region, it gathered surrounding context and used an LLM to provide **definitions, sentence simplifications, and local summaries**. The assistance was presented in a non-intrusive interface so that support could be helpful without fully interrupting the user’s reading process.

## Key results

We evaluated the system with ESL users reading passages of different difficulty levels. The most meaningful gains appeared on harder texts.

For **complex passages**, average comprehension improved from **8.64 to 9.18** out of 10, while average reading time dropped from **517.7 seconds to 471.0 seconds**. Dwell time also decreased substantially, from **96.45 seconds to 58.36 seconds**, suggesting lower cognitive load during reading.

For **simple passages**, the comprehension gain was smaller, moving from **9.82 to 9.91**, which suggests a ceiling effect. Even so, reading time still improved from **249.8 seconds to 218.7 seconds**, showing that the assistant could make reading more efficient even when the text was already relatively easy.

We also observed clear improvements in gaze-based difficulty indicators. On complex passages, **all participants** exceeded five regressions without assistance, but with GPA only **27.3%** did. On simple passages, mean regression count decreased from **2.82 to 1.64**. In the user study, **definitions and simplifications** were rated as the most useful features, while overly frequent assistance could sometimes feel intrusive.

## My contribution

This was a team project, but my main contribution was developing a **working demo page for the gaze-tracking module**.

The hardest engineering problem was making the gaze output actually usable in the document interface. In particular, I worked on **overlaying the detected gaze on top of the document correctly** and then **locating the current gaze position with respect to the entire document**, not just the visible screen region. That required bridging multiple coordinate systems at once: the webcam-based gaze estimate, the rendered browser view, and the document’s global reading position.

That part mattered because the system could only provide relevant assistance if it knew where the user was looking in document space. In practice, this mapping step was what made it possible to connect raw gaze predictions to paragraph-level context, reading progression, and downstream support triggers.

## Why this project matters

This project gave me hands-on experience with building a user-facing ML/HCI system where the hardest problems were not only about model accuracy, but also about **real-time integration**. A gaze model by itself is not enough; the real challenge is turning noisy gaze estimates into reliable document context that a downstream AI assistant can use.

More broadly, it reinforced an important lesson for me: in interactive ML systems, the value often comes from the interface layer that connects sensing, representation, and action. Getting the gaze-to-document alignment right was what made the rest of the system possible.
