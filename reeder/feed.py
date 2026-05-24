"""RSS podcast feed generation from completed jobs."""

import json
from datetime import datetime
from email.utils import format_datetime
from pathlib import Path
from xml.etree.ElementTree import Element, ElementTree, SubElement, indent

from reeder.config import get_paths, load_config


def load_completed_jobs(done_dir: Path) -> list[dict]:
    """Load all successfully completed jobs, sorted newest first."""
    jobs = []
    for job_path in done_dir.glob("*.json"):
        if job_path.name.startswith("FAILED-"):
            continue

        with open(job_path) as f:
            job = json.load(f)

        if "_completed" in job:
            jobs.append(job)

    def get_completion_time(j):
        meta = j["_completed"]
        return meta.get("completed", meta.get("timestamp", ""))

    jobs.sort(key=get_completion_time, reverse=True)
    return jobs


def get_mime_type(filename: str) -> str:
    """Get MIME type for audio file."""
    if filename.endswith(".opus"):
        return "audio/opus"
    elif filename.endswith(".mp3"):
        return "audio/mpeg"
    elif filename.endswith(".m4a"):
        return "audio/mp4"
    else:
        return "audio/mpeg"


def format_duration(seconds: int) -> str:
    """Format duration as HH:MM:SS or MM:SS."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes}:{secs:02d}"


def generate_feed(jobs: list[dict], config: dict, paths: dict) -> Element:
    """Generate RSS 2.0 podcast feed XML."""
    feed_config = config["feed"]
    base_url = feed_config["base_url"].rstrip("/")

    rss = Element("rss")
    rss.set("version", "2.0")
    rss.set("xmlns:itunes", "http://www.itunes.com/dtds/podcast-1.0.dtd")
    rss.set("xmlns:podcast", "https://podcastindex.org/namespace/1.0")

    channel = SubElement(rss, "channel")

    SubElement(channel, "title").text = feed_config["title"]
    SubElement(channel, "description").text = feed_config["description"]
    SubElement(channel, "language").text = feed_config.get("language", "en-us")
    SubElement(channel, "link").text = base_url

    # iTunes metadata
    SubElement(channel, "{http://www.itunes.com/dtds/podcast-1.0.dtd}author").text = feed_config.get("author", "Reeder")
    SubElement(channel, "{http://www.itunes.com/dtds/podcast-1.0.dtd}explicit").text = "false"

    category = SubElement(channel, "{http://www.itunes.com/dtds/podcast-1.0.dtd}category")
    category.set("text", "Technology")

    # Feed image
    if "image" in feed_config:
        image_url = f"{base_url}/{feed_config['image']}"
        itunes_image = SubElement(channel, "{http://www.itunes.com/dtds/podcast-1.0.dtd}image")
        itunes_image.set("href", image_url)

        image = SubElement(channel, "image")
        SubElement(image, "url").text = image_url
        SubElement(image, "title").text = feed_config["title"]
        SubElement(image, "link").text = base_url

    # Episodes
    for job in jobs:
        completed = job["_completed"]
        audio_file = completed["audio_file"]
        audio_url = f"{base_url}/audio/{audio_file}"

        item = SubElement(channel, "item")
        SubElement(item, "title").text = job.get("title", "Untitled")
        SubElement(item, "guid").text = completed["guid"]

        timestamp_str = completed.get("completed", completed.get("timestamp"))
        timestamp = datetime.fromisoformat(timestamp_str)
        SubElement(item, "pubDate").text = format_datetime(timestamp)

        description = job.get("description", "")
        if not description and job.get("type") == "url":
            description = f"Audio version of {job['url']}"
        SubElement(item, "description").text = description

        source_url = job.get("source_url") or job.get("url")
        if source_url:
            SubElement(item, "link").text = source_url

        enclosure = SubElement(item, "enclosure")
        enclosure.set("url", audio_url)
        enclosure.set("length", str(completed["audio_size"]))
        enclosure.set("type", get_mime_type(audio_file))

        duration = completed.get("duration_seconds", 0)
        SubElement(item, "{http://www.itunes.com/dtds/podcast-1.0.dtd}duration").text = str(duration)
        SubElement(item, "{http://www.itunes.com/dtds/podcast-1.0.dtd}explicit").text = "false"

    return rss


def update_feed(config: dict | None = None):
    """Generate and write the RSS feed. Main entry point."""
    if config is None:
        config = load_config()
    paths = get_paths(config)

    jobs = load_completed_jobs(paths["done"])
    print(f"Found {len(jobs)} completed jobs")

    rss = generate_feed(jobs, config, paths)
    indent(rss, space="  ")

    feed_path = paths["www"] / "feed.xml"
    tree = ElementTree(rss)
    tree.write(feed_path, encoding="unicode", xml_declaration=True)
    print(f"Feed written to {feed_path}")
