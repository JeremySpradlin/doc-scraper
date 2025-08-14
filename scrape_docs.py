import argparse
import os
import re
import time
import hashlib
from collections import deque
from dataclasses import dataclass
from typing import Optional, Set, Tuple, Dict
from urllib.parse import urljoin, urlparse, urldefrag, urlunparse, urlunparse

import requests
from bs4 import BeautifulSoup
from bs4 import UnicodeDammit
from markdownify import markdownify as md
from readability import Document
from slugify import slugify
from tqdm import tqdm


DEFAULT_USER_AGENT = (
    "doc-scraper/0.1 (+https://github.com/ErbunTech)"
)


@dataclass
class CrawlConfig:
    start_url: str
    output_dir: str
    scope: str  # 'prefix' or 'domain'
    max_pages: Optional[int]
    delay_seconds: float
    use_readability: bool
    user_agent: str
    timeout_seconds: float


def normalize_url(url: str) -> str:
    url, _ = urldefrag(url)
    parsed = urlparse(url)
    # Normalize: lowercase scheme/host, remove default ports, remove query sorting is left as is
    netloc = parsed.hostname or ""
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"
    normalized = parsed._replace(scheme=parsed.scheme.lower(), netloc=netloc)
    return urlunparse(normalized)


def is_in_scope(candidate_url: str, base: urlparse, scope: str) -> bool:
    parsed = urlparse(candidate_url)
    if parsed.scheme not in ("http", "https"):
        return False
    if parsed.netloc != base.netloc:
        return False
    if scope == "domain":
        return True
    # prefix scope: must start with base.path
    base_path = base.path if base.path.endswith("/") else base.path + "/"
    cand_path = parsed.path if parsed.path.endswith("/") else parsed.path + "/"
    return cand_path.startswith(base_path)


def compute_site_folder_name(start_url: str) -> str:
    parsed = urlparse(start_url)
    netloc_slug = slugify(parsed.netloc)
    path_slug = slugify(parsed.path.strip("/"))
    if path_slug:
        return f"{netloc_slug}-{path_slug}"
    return netloc_slug


def url_to_output_paths(start_base: urlparse, output_root: str, target_url: str) -> Tuple[str, str]:
    parsed = urlparse(target_url)
    # Compute relative path under the site root
    rel_path = parsed.path
    if rel_path.startswith("/"):
        # make it relative to the base path if inside prefix
        base_path = start_base.path
        if rel_path.startswith(base_path):
            rel_path = rel_path[len(base_path):]
        else:
            rel_path = rel_path.lstrip("/")
    rel_path = rel_path.strip("/")

    # Encode query into filename (use short hash to avoid overly long names)
    query_suffix = ""
    if parsed.query:
        query_hash = hashlib.sha1(parsed.query.encode("utf-8")).hexdigest()[:8]
        query_suffix = f"_{query_hash}"

    if rel_path == "":
        md_path = os.path.join(output_root, "index" + query_suffix + ".md")
        assets_dir = os.path.join(output_root, "assets")
        return md_path, assets_dir

    base_name = os.path.basename(rel_path)
    dir_name = os.path.dirname(rel_path)

    root_dir = os.path.join(output_root, dir_name)

    name, ext = os.path.splitext(base_name)
    if ext.lower() in (".html", ".htm", ""):
        if base_name == "" or rel_path.endswith("/"):
            md_path = os.path.join(root_dir, "index" + query_suffix + ".md")
        else:
            md_path = os.path.join(root_dir, name + query_suffix + ".md")
    else:
        # Non-HTML path treated as a page
        md_path = os.path.join(root_dir, name + query_suffix + ".md")

    assets_dir = os.path.join(os.path.dirname(md_path), "assets")
    return md_path, assets_dir


def ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _parse_srcset(srcset_value: str, base_url: str) -> Optional[str]:
    try:
        # Choose the first candidate as a reasonable default
        first_part = srcset_value.split(",")[0].strip()
        url_only = first_part.split(" ")[0]
        return urljoin(base_url, url_only)
    except Exception:
        return None


def save_image(session: requests.Session, image_url: str, assets_dir: str, user_agent: str, timeout: float) -> Optional[str]:
    try:
        headers = {"User-Agent": user_agent}
        response = session.get(image_url, headers=headers, timeout=timeout)
        if response.status_code != 200:
            return None
        content_type = response.headers.get("Content-Type", "").lower()
        if not ("image/" in content_type):
            return None
        os.makedirs(assets_dir, exist_ok=True)
        parsed = urlparse(image_url)
        filename = os.path.basename(parsed.path) or "image"
        if not os.path.splitext(filename)[1]:
            # try to infer from content type
            ext = content_type.split("/")[-1].split(";")[0]
            filename = f"{filename}.{ext}"
        # Prevent collisions
        hash_suffix = hashlib.sha1(image_url.encode("utf-8")).hexdigest()[:8]
        name, ext = os.path.splitext(filename)
        local_name = f"{slugify(name)}-{hash_suffix}{ext}"
        local_path = os.path.join(assets_dir, local_name)
        with open(local_path, "wb") as f:
            f.write(response.content)
        return local_path
    except Exception:
        return None


def discover_links(soup: BeautifulSoup, base_url: str, start_base: urlparse, scope: str) -> Set[str]:
    discovered: Set[str] = set()
    for a in soup.find_all("a"):
        href = a.get("href")
        if not href or href.startswith("mailto:") or href.startswith("tel:"):
            continue
        abs_href = urljoin(base_url, href)
        abs_href, _ = urldefrag(abs_href)
        if is_in_scope(abs_href, start_base, scope):
            discovered.add(abs_href)
    return discovered


def rewrite_links_and_download_assets(
    session: requests.Session,
    soup: BeautifulSoup,
    start_base: urlparse,
    current_page_dir: str,
    output_root: str,
    current_page_url: str,
    config: CrawlConfig,
) -> Tuple[Set[str], BeautifulSoup]:
    discovered: Set[str] = set()

    # Images (handle src and srcset)
    for img in soup.find_all("img"):
        src = img.get("src")
        srcset = img.get("srcset")
        candidate_url: Optional[str] = None
        if srcset:
            candidate_url = _parse_srcset(srcset, current_page_url)
        if not candidate_url and src:
            candidate_url = urljoin(current_page_url, src)
        if not candidate_url:
            continue
        local_img = save_image(
            session=session,
            image_url=candidate_url,
            assets_dir=os.path.join(current_page_dir, "assets"),
            user_agent=config.user_agent,
            timeout=config.timeout_seconds,
        )
        if local_img:
            rel = os.path.relpath(local_img, current_page_dir)
            img["src"] = rel
            if "srcset" in img.attrs:
                del img["srcset"]

    # Anchors: rewrite internal links to local paths; also collect for discovery
    for a in soup.find_all("a"):
        href = a.get("href")
        if not href:
            continue
        if href.startswith("mailto:") or href.startswith("tel:"):
            continue
        abs_href = urljoin(current_page_url, href)
        abs_href, frag = urldefrag(abs_href)
        if is_in_scope(abs_href, start_base, config.scope):
            discovered.add(abs_href)
            md_path, _ = url_to_output_paths(start_base, output_root, abs_href)
            rel_link = os.path.relpath(md_path, current_page_dir)
            if frag:
                rel_link = f"{rel_link}#{frag}"
            a["href"] = rel_link
        else:
            a["target"] = a.get("target") or "_blank"

    return discovered, soup


def fetch_page(session: requests.Session, url: str, config: CrawlConfig) -> Optional[str]:
    try:
        headers = {"User-Agent": config.user_agent}
        response = session.get(url, headers=headers, timeout=config.timeout_seconds)
        if response.status_code != 200:
            return None
        content_type = response.headers.get("Content-Type", "").lower()
        if "html" not in content_type:
            return None
        # Robust decoding
        dammit = UnicodeDammit(response.content)
        html_text = dammit.unicode_markup
        return html_text
    except Exception:
        return None


def html_to_clean_html(html: str, use_readability: bool) -> str:
    content_html = html
    if use_readability:
        try:
            doc = Document(html)
            content_html = doc.summary(html_partial=True)
        except Exception:
            content_html = html
    soup = BeautifulSoup(content_html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return str(soup)


def crawl_and_save(config: CrawlConfig) -> None:
    start_base = urlparse(config.start_url)

    site_folder = compute_site_folder_name(config.start_url)
    output_root = os.path.abspath(os.path.join(config.output_dir, site_folder))
    os.makedirs(output_root, exist_ok=True)

    queue: deque[str] = deque()
    visited: Set[str] = set()

    normalized_start = normalize_url(config.start_url)
    queue.append(normalized_start)

    session = requests.Session()

    progress = tqdm(total=config.max_pages or 0, unit="page", disable=(config.max_pages is None))

    saved_count = 0

    while queue:
        if config.max_pages is not None and saved_count >= config.max_pages:
            break
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)

        raw_html = fetch_page(session, current, config)
        if raw_html is None:
            continue

        # Build soup from full HTML for link discovery and title fallback
        raw_soup = BeautifulSoup(raw_html, "lxml")

        # Clean/Readability for content extraction
        cleaned_html = html_to_clean_html(raw_html, config.use_readability)
        content_soup = BeautifulSoup(cleaned_html, "lxml")

        md_path, assets_dir = url_to_output_paths(start_base, output_root, current)
        ensure_parent_dir(md_path)
        page_dir = os.path.dirname(md_path)

        # Rewrite links and download assets within the content soup
        discovered_from_content, rewritten_soup = rewrite_links_and_download_assets(
            session=session,
            soup=content_soup,
            start_base=start_base,
            current_page_dir=page_dir,
            output_root=output_root,
            current_page_url=current,
            config=config,
        )

        # Discover additional links from the full HTML to avoid missing nav links pruned by readability
        discovered_from_full = discover_links(raw_soup, current, start_base, config.scope)

        # Merge discovered sets
        discovered = discovered_from_content.union(discovered_from_full)

        # Enqueue new links
        for d in discovered:
            d_norm = normalize_url(d)
            if d_norm not in visited:
                queue.append(d_norm)

        # Convert to markdown
        markdown = md(str(rewritten_soup), heading_style="ATX")

        # Title handling: prefer h1 from content; fallback to h1/title from raw
        title_tag = rewritten_soup.find("h1") or raw_soup.find("h1") or raw_soup.find("title")
        if title_tag:
            title_text = title_tag.get_text(strip=True)
            if title_text and not markdown.lstrip().startswith("# "):
                markdown = f"# {title_text}\n\n" + markdown

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown)

        saved_count += 1
        if config.max_pages is not None:
            progress.update(1)

        if config.delay_seconds > 0:
            time.sleep(config.delay_seconds)

    progress.close()


def parse_args() -> CrawlConfig:
    parser = argparse.ArgumentParser(description="Scrape documentation sites into local Markdown with images.")
    parser.add_argument("url", help="Entry URL for the documentation site")
    parser.add_argument(
        "--output-dir",
        default="./scraped-docs",
        help="Directory where the site folder will be created (default: ./scraped-docs)",
    )
    parser.add_argument(
        "--scope",
        choices=["prefix", "domain"],
        default="prefix",
        help="Crawl scope: 'prefix' restricts to the starting path; 'domain' allows the whole domain (default: prefix)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum number of pages to save (default: unlimited)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Delay in seconds between requests (default: 0)",
    )
    parser.add_argument(
        "--no-readability",
        action="store_true",
        help="Disable readability extraction (keep full page)",
    )
    parser.add_argument(
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help="User-Agent header to use for requests",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="Request timeout in seconds (default: 20)",
    )

    args = parser.parse_args()

    config = CrawlConfig(
        start_url=args.url,
        output_dir=args.output_dir,
        scope=args.scope,
        max_pages=args.max_pages,
        delay_seconds=args.delay,
        use_readability=not args.no_readability,
        user_agent=args.user_agent,
        timeout_seconds=args.timeout,
    )

    return config


def main() -> None:
    config = parse_args()
    crawl_and_save(config)


if __name__ == "__main__":
    main()
