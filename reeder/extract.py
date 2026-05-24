"""Text extraction from URLs and raw text."""

import subprocess
from urllib.parse import urlparse


def extract_text_trafilatura(url: str) -> str:
    """Extract article text using trafilatura."""
    import trafilatura

    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise ValueError(f"Failed to fetch URL: {url}")

    text = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=False,
    )
    if not text:
        raise ValueError(f"Failed to extract text from: {url}")

    return text


def extract_text_pup(url: str, selector: str) -> str:
    """Extract text using curl + pup with a CSS selector."""
    result = subprocess.run(
        ["curl", "-sL", url],
        capture_output=True,
        text=True,
        check=True,
    )
    html = result.stdout

    result = subprocess.run(
        ["pup", selector],
        input=html,
        capture_output=True,
        text=True,
        check=True,
    )
    text = result.stdout.strip()

    if not text:
        raise ValueError(f"No text found with selector '{selector}' at {url}")

    return text


def extract_text_readability(url: str) -> str:
    """Extract text using readability-cli."""
    result = subprocess.run(
        ["readable", url, "-p"],
        capture_output=True,
        text=True,
        check=True,
    )
    text = result.stdout.strip()

    if not text:
        raise ValueError(f"Failed to extract text with readability from: {url}")

    return text


def extract_text(job: dict, config: dict) -> str:
    """Extract text based on job type and extractor settings."""
    if job["type"] == "text":
        return job["text"]

    url = job["url"]
    extractor = job.get("extractor", config["extractors"]["default"])

    # Check for site-specific extractor
    if extractor == "auto":
        domain = urlparse(url).netloc
        site_extractors = config.get("extractors", {}).get("sites", {})
        if domain in site_extractors:
            extractor = {"pup": site_extractors[domain]}
        else:
            extractor = "trafilatura"

    if isinstance(extractor, dict) and "pup" in extractor:
        return extract_text_pup(url, extractor["pup"])
    elif extractor == "readability":
        return extract_text_readability(url)
    else:  # trafilatura (default)
        return extract_text_trafilatura(url)
