import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import tailwind from "@astrojs/tailwind";
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

const remarkPlugins = [remarkMath];
const rehypePlugins = [rehypeKatex];

// https://astro.build/config
export default defineConfig({
  site: 'https://giwone1330.github.io',
  markdown: {
    remarkPlugins,
    rehypePlugins,
  },
  integrations: [mdx({ remarkPlugins, rehypePlugins }), tailwind()]
});