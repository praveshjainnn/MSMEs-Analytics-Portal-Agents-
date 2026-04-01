import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import time

_cache = {}
CACHE_TTL = 3600 * 4 # 4 hours

def fetch_district_news(district, state="India", limit=10):
    """
    Fetches recent news headlines for a district focusing on business/economy via Google News RSS.
    Returns a list of dicts: [{'title': ..., 'link': ..., 'date': ...}, ...]
    """
    ck = f"news_{district}_{state}"
    if ck in _cache and time.time() - _cache[ck][1] < CACHE_TTL:
        return _cache[ck][0]

    # Construct query: "District" AND ("MSME" OR "business" OR "industry" OR "economy" OR "investment" OR "infrastructure")
    query = f'"{district}" ("MSME" OR "business" OR "industry" OR "economy" OR "investment" OR "infrastructure" OR "employment")'
    encoded_query = urllib.parse.quote(query)
    
    # We use Google News India RSS
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "application/rss+xml, application/xml"
    }
    
    articles = []
    try:
        req = urllib.request.Request(url, headers=headers)
        resp = urllib.request.urlopen(req, timeout=12)
        xml_data = resp.read()
        root = ET.fromstring(xml_data)
        
        for item in root.findall('.//item')[:limit]:
            title = item.find('title').text if item.find('title') is not None else ""
            link = item.find('link').text if item.find('link') is not None else ""
            pub_date = item.find('pubDate').text if item.find('pubDate') is not None else ""
            
            # Clean up title (Google news often appends " - Publisher Name" at the end)
            if " - " in title:
                title = " - ".join(title.split(" - ")[:-1])
                
            articles.append({
                "title": title.strip(),
                "link": link,
                "date": pub_date
            })
            
        _cache[ck] = (articles, time.time())
        return articles
    except Exception as e:
        print(f"[SentimentScraper] Error fetching news for {district}: {e}")
        return []

def format_news_for_prompt(articles):
    """Formats the list of articles into a structured prompt for Ollama."""
    if not articles:
        return "No recent relevant business or economic news found for this district."
    
    lines = []
    for i, art in enumerate(articles, 1):
        lines.append(f"Headline {i}: {art['title']}")
        # We don't need to pass the date or link to the LLM for sentiment extraction, keeping context window light.
    
    return "\n".join(lines)
