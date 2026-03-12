import { getCollection } from "astro:content";
import createSlug from "../lib/createSlug";

function escapeXml(value) {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

function toUrl(site, path) {
  return new URL(path, site).toString();
}

function withTrailingSlash(path) {
  return path.endsWith("/") ? path : `${path}/`;
}

export async function GET(context) {
  const site = context.site ?? new URL("https://giwone1330.github.io");

  const blog = await getCollection("blog");
  const projects = await getCollection("projects");
  const publications = await getCollection("publications");
  const store = await getCollection("store");

  const staticPages = [
    "/",
    "/blog/",
    "/projects/",
    "/publications/",
    "/store/",
    "/cv/",
    "/services/",
    "/rss.xml",
  ];

  const blogPages = blog.map((entry) => withTrailingSlash(`/blog/${createSlug(entry.data.title, entry.slug)}`));
  const projectPages = projects.map((entry) => withTrailingSlash(`/projects/${createSlug(entry.data.title, entry.slug)}`));
  const publicationPages = publications.map((entry) => withTrailingSlash(`/publications/${createSlug(entry.data.title, entry.slug)}`));
  const storePages = store.map((entry) => withTrailingSlash(`/store/${entry.slug}`));

  const urls = [...staticPages, ...blogPages, ...projectPages, ...publicationPages, ...storePages]
    .map((path) => toUrl(site, path))
    .sort();

  const xml = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${urls.map((url) => `  <url><loc>${escapeXml(url)}</loc></url>`).join("\n")}
</urlset>`;

  return new Response(xml, {
    headers: {
      "Content-Type": "application/xml; charset=utf-8",
    },
  });
}