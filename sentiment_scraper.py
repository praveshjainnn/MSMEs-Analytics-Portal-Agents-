"""
sentiment_scraper.py  —  Live Media Pulse Module for MSME Analytics Dashboard
================================================================================
PURPOSE:
    This module acts as the "ground-level intelligence" layer of the dashboard.
    It fetches real-time business/economic news headlines for any Indian district
    from Google News RSS, then formats them into a structured context block
    that can be fed into the local Ollama LLM for sentiment analysis.

HOW IT WORKS:
    1. `fetch_district_news()` builds a targeted search query, hits the
       Google News RSS feed, parses the XML response, and returns a list
       of article dicts (title, link, date).
    2. `format_news_for_prompt()` converts that list into a plain-text
       numbered headline block optimised for the LLM's context window.
    3. An in-process cache (TTL: 4 hours) avoids hammering the RSS endpoint
       for the same district repeatedly.

DEPENDENCIES:
    - urllib.request  : standard-library HTTP client (no external packages)
    - xml.etree.ElementTree : standard-library RSS/XML parser
    - time            : for cache TTL management
"""

import urllib.request   # Standard-library HTTP client — no pip install needed
import urllib.parse     # Used to URL-encode the search query string safely
import xml.etree.ElementTree as ET  # Parses the XML returned by the Google News RSS feed
import time             # Used to track cache timestamps for TTL expiry


# ─────────────────────────────────────────────────────────────────────────────
#  IN-PROCESS CACHE
# ─────────────────────────────────────────────────────────────────────────────
# A simple dict that maps a cache key (district + state) to a tuple of
# (articles_list, timestamp). This avoids redundant RSS fetches when the
# user queries the same district multiple times within the TTL window.
_cache = {}

# Cache Time-To-Live: 4 hours (in seconds).
# News freshness matters less than performance here — 4 hours is a
# reasonable trade-off between staleness and server load.
CACHE_TTL = 3600 * 4  # = 14,400 seconds = 4 hours


# ─────────────────────────────────────────────────────────────────────────────
#  FUNCTION: fetch_district_news
# ─────────────────────────────────────────────────────────────────────────────
def fetch_district_news(district, state="India", limit=10):
    """
    Fetches recent news headlines for a district focusing on
    business/economy via Google News RSS.

    PARAMETERS:
        district (str) : The name of the district to search for (e.g., "Pune")
        state    (str) : The state name, used as additional search context.
                         Defaults to "India" for a broad search.
        limit    (int) : Maximum number of articles to return. Default: 10.

    RETURNS:
        list[dict] : A list of article dictionaries, each containing:
                     - 'title' (str) : The cleaned headline text
                     - 'link'  (str) : URL to the original news article
                     - 'date'  (str) : Publication date string from RSS

    RETURNS [] on network error or parse failure (never raises an exception).

    HOW THE QUERY WORKS:
        The query forces Google News to only return articles that:
          - Mention the specific district name exactly (quoted string)
          - AND contain at least one economic/business keyword
        This dramatically improves the signal-to-noise ratio of results.
    """

    # Build a unique cache key from the district and state names
    ck = f"news_{district}_{state}"

    # Check cache — return immediately if we have a fresh result
    if ck in _cache and time.time() - _cache[ck][1] < CACHE_TTL:
        return _cache[ck][0]

    # ── Construct the targeted Google News search query ──
    # The query uses AND logic: exact district name PLUS at least one
    # of the economic keywords. This filters out political/sports news.
    query = f'"{district}" ("MSME" OR "business" OR "industry" OR "economy" OR "investment" OR "infrastructure" OR "employment")'

    # URL-encode the query string so special characters (quotes, spaces, OR)
    # are safely transmitted in the URL
    encoded_query = urllib.parse.quote(query)

    # Google News RSS endpoint — 'hl=en-IN' sets Hindi+English (India),
    # 'gl=IN' geo-targets India, 'ceid=IN:en' sets country+language edition
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"

    # Browser-style headers to avoid being blocked by Google's bot detection
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "application/rss+xml, application/xml"
    }

    articles = []  # Will hold the parsed article dicts

    try:
        # Make the HTTP GET request to Google News RSS
        req = urllib.request.Request(url, headers=headers)
        resp = urllib.request.urlopen(req, timeout=12)  # 12-second timeout

        # Read and parse the raw XML response
        xml_data = resp.read()
        root = ET.fromstring(xml_data)  # Parse XML into an ElementTree

        # Iterate over all <item> elements (each = one news article)
        # Slice to [limit] to cap the number returned
        for item in root.findall('.//item')[:limit]:

            # Safely extract each sub-element's text, defaulting to "" if missing
            title    = item.find('title').text   if item.find('title')   is not None else ""
            link     = item.find('link').text    if item.find('link')    is not None else ""
            pub_date = item.find('pubDate').text if item.find('pubDate') is not None else ""

            # Google News appends " - Publisher Name" to every title
            # (e.g., "New MSME Hub Opens in Pune - The Hindu")
            # We strip the last " - Publisher" segment to get the clean headline
            if " - " in title:
                title = " - ".join(title.split(" - ")[:-1])

            articles.append({
                "title": title.strip(),
                "link":  link,
                "date":  pub_date
            })

        # Store the fresh result in cache with the current timestamp
        _cache[ck] = (articles, time.time())
        return articles

    except Exception as e:
        # Log the error but NEVER crash — return an empty list gracefully
        # The calling callback handles the empty-list case with a warning message
        print(f"[SentimentScraper] Error fetching news for {district}: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
#  FUNCTION: format_news_for_prompt
# ─────────────────────────────────────────────────────────────────────────────
def format_news_for_prompt(articles):
    """
    Converts a list of article dicts into a numbered plain-text block
    that can be directly injected into an Ollama LLM prompt.

    DESIGN RATIONALE:
        - We intentionally exclude 'link' and 'date' from the prompt.
          The LLM only needs the headline text for sentiment extraction.
          Passing URLs would waste precious context-window tokens.
        - Numbered headlines ("Headline 1:", "Headline 2:", etc.) help
          the LLM refer to specific articles in its analysis.

    PARAMETERS:
        articles (list[dict]) : Output of `fetch_district_news()`

    RETURNS:
        str : A formatted multi-line string like:
              "Headline 1: New MSME hub opens in Pune
               Headline 2: Textile industry faces power shortage..."
              Or a fallback message string if the list is empty.
    """

    # Handle the case where no articles were found or fetching failed
    if not articles:
        return "No recent relevant business or economic news found for this district."

    lines = []
    # Enumerate from 1 for human-readable numbering (e.g., "Headline 1:")
    for i, art in enumerate(articles, 1):
        lines.append(f"Headline {i}: {art['title']}")
        # NOTE: Deliberately omitting art['date'] and art['link'] here
        # to keep the LLM context window lean and focused on text only.

    # Join all headline lines into a single newline-separated string
    return "\n".join(lines)
