---
title: "Initial post"
description: "Brief description of how this homepage was made, and note to self on how to make future updates."
pubDate: "Mar 13 2026"
heroImage: "/post_img.webp"
badge: "Demo badge"
tags: ["astro","tailwindcss","vite"]
---

## Building My Portfolio with Astro

This portfolio is built using the **Astrofy** template, an open-source
portfolio framework built on Astro.

Instead of building a static site from scratch, I started from Astrofy
to leverage a solid baseline architecture while focusing on
customization and content. The template provides a clean component
structure, a blog system, and Tailwind integration out of the box.

This post describes the **technology stack and architecture** behind the
site, along with the development and deployment workflow.

Template repository:\
https://github.com/manuelernestog/astrofy

------------------------------------------------------------------------

## System Overview

The site is implemented as a **static-first web application**. All pages
are generated at build time and deployed as static assets.

**Core technologies**

-   Astro (static site framework)
-   Tailwind CSS (utility-first styling)
-   DaisyUI (component layer on Tailwind)
-   Markdown content collections
-   GitHub Pages (hosting)
-   GitHub Actions (CI/CD)

This architecture keeps the runtime footprint minimal while maintaining
a modular codebase.

------------------------------------------------------------------------

## Framework: Astro

The site is built with **Astro**, a modern web framework designed around
server-first rendering and static generation.

Astro compiles components and content into **static HTML during the
build step**, which significantly reduces client-side JavaScript.

Benefits of this approach include:

-   faster initial page loads
-   smaller JavaScript bundles
-   improved SEO and accessibility
-   straightforward hosting on static infrastructure

Pages are composed from `.astro` components.

Example:

    src/components/
      Header.astro
      Footer.astro
      ProjectCard.astro

Each component encapsulates markup, styling, and logic while remaining
lightweight at runtime.

------------------------------------------------------------------------

## Styling System

The UI layer is implemented using **Tailwind CSS** combined with
**DaisyUI**.

## Tailwind CSS

Tailwind provides utility classes for layout, spacing, typography, and
responsiveness.

Example:

``` html
<div class="flex items-center gap-4 p-6 rounded-xl bg-base-100">
```

Using utilities instead of custom CSS helps maintain consistent design
patterns across the site while reducing stylesheet complexity.

## DaisyUI

DaisyUI builds on top of Tailwind and provides a set of composable UI
primitives such as:

-   navigation bars
-   cards
-   buttons
-   layouts

This avoids writing repetitive styling logic while still keeping
everything within the Tailwind ecosystem.

------------------------------------------------------------------------

## Content Management

Content such as blog posts is written in **Markdown** and managed using
Astro's content collections.

Directory structure:

    src/content/blog/

Each post includes frontmatter metadata:

``` markdown
---
title: "Example Post"
description: "Post summary"
date: 2026-03-13
---
```

During the build process Astro converts Markdown into static HTML pages.

This keeps the content workflow simple while ensuring everything remains
version-controlled.

------------------------------------------------------------------------

## Project Structure

The repository follows the typical Astro layout:

    src/
      components/
      layouts/
      pages/
      content/
      styles/
    public/
    astro.config.mjs
    tailwind.config.cjs
    package.json

Key directories:

**components**\
Reusable UI components used across pages.

**layouts**\
Shared page structures such as headers and metadata wrappers.

**pages**\
File-based routing for site URLs.

**content**\
Markdown-based content collections.

**public**\
Static assets including images and icons.

------------------------------------------------------------------------

## Development Workflow

After cloning the repository, the local development process is
straightforward.

Install dependencies:

``` bash
npm install
```

Start the development server:

``` bash
npm run dev
```

Astro launches a local development server with hot module reload for
both components and content changes.

------------------------------------------------------------------------

## Static Build Pipeline

Production builds are generated using:

``` bash
npm run build
```

Astro compiles the site into static output located in:

    dist/

The generated output contains:

-   static HTML
-   optimized CSS
-   minimal JavaScript
-   bundled assets

Because the site is fully static, it can be deployed on any CDN-backed
hosting platform.

------------------------------------------------------------------------

## Deployment

The site is deployed through **GitHub Pages** using an automated CI
workflow.

Deployment flow:

1.  Push changes to the repository
2.  GitHub Actions installs dependencies
3.  Astro builds the static site
4.  The `dist/` directory is deployed to GitHub Pages

This pipeline ensures builds are reproducible and deployment is fully
automated.

The site becomes accessible at:

    https://username.github.io/repository-name/

------------------------------------------------------------------------

## Performance Considerations

Astro's architecture allows the site to remain lightweight without
additional optimization work.

Key performance characteristics:

**Static rendering**

All pages are generated during the build step.

**Minimal client-side JavaScript**

JavaScript is only shipped when explicitly required.

**CDN-backed hosting**

GitHub Pages distributes assets through a global CDN.

Together these design choices keep the site responsive while keeping
operational complexity low.

------------------------------------------------------------------------

<!-- ## Closing Notes

Although the site itself is relatively simple, the goal was to build it
using tools that emphasize:

-   modular architecture
-   minimal runtime overhead
-   automated deployment
-   maintainable content workflows

Astro provides a clean and efficient framework for this type of project,
and Astrofy serves as a solid starting point for structuring the
codebase.

------------------------------------------------------------------------ -->

Template repository:\
https://github.com/manuelernestog/astrofy
