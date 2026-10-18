import html
import json
import os
import re
import smtplib
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import requests
from openai import OpenAI


KEYWORDS = [
    "tactile sensor", "flexible sensor", "electronic skin", "ionogel",
    "ionic gel", "ionic liquid", "electric double layer capacitor",
    "EDL capacitor", "3D force sensing", "3D force sensor", "PVDF-TrFE",
    "piezoelectric sensor", "piezoelectric", "triboelectric",
    "wearable electronics", "ionic capacitor", "iontronic",]

JOURNALS = [
    "Advanced Materials", "Advanced Functional Materials", "ACS Nano",
    "Nano Energy", "Nature", "Nature Materials", "Nature Electronics",
    "Nature Nanotechnology", "Nature Communications",
    "Communications Materials", "Communications Engineering",
    "npj Flexible Electronics",
    "Small", "Advanced Science", "Chemical Engineering Journal",
]

# RSS is best-effort. Crossref and Semantic Scholar cover every journal in the
# whitelist; publisher feeds add newly posted articles before all indexes sync.
RSS_FEEDS = {
    "Nature RSS": [
        ("Nature", "https://www.nature.com/nature.rss"),
        ("Nature Materials", "https://www.nature.com/nmat.rss"),
        ("Nature Electronics", "https://www.nature.com/natelectron.rss"),
        ("Nature Nanotechnology", "https://www.nature.com/nnano.rss"),
        ("Nature Communications", "https://www.nature.com/ncomms.rss"),
        ("Communications Materials", "https://www.nature.com/commsmat.rss"),
        ("Communications Engineering", "https://www.nature.com/commseng.rss"),
        ("npj Flexible Electronics", "https://www.nature.com/npjflexelectron.rss"),
    ],
    "ACS RSS": [
        ("ACS Nano", "https://pubs.acs.org/action/showFeed?type=etoc&feed=rss&jc=ancac3"),
    ],
    "Wiley RSS": [
        ("Advanced Materials", "https://advanced.onlinelibrary.wiley.com/feed/15214095/most-recent"),
        ("Advanced Functional Materials", "https://advanced.onlinelibrary.wiley.com/feed/16163028/most-recent"),
        ("Small", "https://small-journal.com/feed/16136829/most-recent"),
        ("Advanced Science", "https://advanced.onlinelibrary.wiley.com/feed/21983844/most-recent"),
    ],
}

LOOKBACK_DAYS = int(os.environ.get("LOOKBACK_DAYS", "7"))
MAX_RESULTS = int(os.environ.get("MAX_RESULTS", "10"))
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api-inference.modelscope.cn/v1"
DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-V3.2-Exp"
SEMANTIC_SCHOLAR_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
SENDER_EMAIL = os.environ.get("SENDER_EMAIL")
SENDER_PASSWORD = os.environ.get("SENDER_PASSWORD")
RECEIVER_EMAIL = os.environ.get("RECEIVER_EMAIL")
SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.163.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "465"))

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "FlexibleSensorDaily/1.0 (mailto:luxi0131@163.com)",
    "Accept": "application/json",
})


def clean_text(value):
    value = re.sub(r"<[^>]+>", " ", value or "")
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def normalize(value):
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


JOURNAL_ALIASES = {normalize(name): name for name in JOURNALS}
JOURNAL_ALIASES.update({
    "adv mater": "Advanced Materials",
    "adv funct mater": "Advanced Functional Materials",
    "nat commun": "Nature Communications",
    "npj flex electron": "npj Flexible Electronics",
    "adv sci weinh": "Advanced Science",
    "chem eng j": "Chemical Engineering Journal",
})


def canonical_journal(value):
    key = normalize(value)
    if key in JOURNAL_ALIASES:
        return JOURNAL_ALIASES[key]
    for alias, official in JOURNAL_ALIASES.items():
        if alias and (alias in key or key in alias):
            return official
    return None


def matched_keywords(title, abstract):
    haystack = normalize(f"{title} {abstract}")
    return [keyword for keyword in KEYWORDS if normalize(keyword) in haystack]


def parse_date(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()
    except ValueError:
        return None


def crossref_date(item):
    for field in ("published-online", "published-print", "published", "issued", "created"):
        parts = item.get(field, {}).get("date-parts", [[]])[0]
        if parts:
            try:
                return date(*((parts + [1, 1])[:3]))
            except (TypeError, ValueError):
                pass
    return None


def fetch_crossref():
    print("Fetching Crossref...")
    start = (date.today() - timedelta(days=LOOKBACK_DAYS)).isoformat()
    papers = []
    for keyword in KEYWORDS:
        params = {
            "query.bibliographic": keyword,
            "filter": f"from-pub-date:{start},type:journal-article",
            "select": "DOI,title,container-title,abstract,URL,author,published,published-online,published-print,issued,created",
            "rows": 20, "sort": "published", "order": "desc",
            "mailto": "luxi0131@163.com",
        }
        try:
            response = SESSION.get("https://api.crossref.org/works", params=params, timeout=30)
            response.raise_for_status()
            for item in response.json()["message"]["items"]:
                journal = canonical_journal((item.get("container-title") or [""])[0])
                title = clean_text((item.get("title") or [""])[0])
                abstract = clean_text(item.get("abstract", ""))
                matches = matched_keywords(title, abstract)
                if not journal or not matches:
                    continue
                papers.append({
                    "title": title, "journal": journal,
                    "doi": (item.get("DOI") or "").lower(),
                    "url": item.get("URL") or "", "abstract": abstract,
                    "published": crossref_date(item),
                    "authors": ", ".join(
                        " ".join(filter(None, [a.get("given"), a.get("family")]))
                        for a in item.get("author", [])
                    ),
                    "keywords": matches, "sources": ["Crossref"],
                    "citation_count": 0,
                })
        except Exception as exc:
            print(f"Crossref warning for {keyword}: {exc}")
        time.sleep(0.08)
    return papers


def fetch_semantic_scholar():
    print("Fetching Semantic Scholar...")
    papers = []
    headers = {"x-api-key": SEMANTIC_SCHOLAR_API_KEY} if SEMANTIC_SCHOLAR_API_KEY else {}
    fields = "title,abstract,venue,publicationDate,externalIds,url,authors,citationCount"
    current_year = date.today().year
    for keyword in KEYWORDS:
        try:
            response = SESSION.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={"query": keyword, "limit": 20, "fields": fields, "year": f"{current_year - 1}-"},
                headers=headers, timeout=30,
            )
            if response.status_code == 429:
                print("Semantic Scholar rate limit reached; continuing with other sources")
                break
            response.raise_for_status()
            for item in response.json().get("data", []):
                journal = canonical_journal(item.get("venue", ""))
                title = clean_text(item.get("title", ""))
                abstract = clean_text(item.get("abstract", ""))
                matches = matched_keywords(title, abstract)
                published = parse_date(item.get("publicationDate"))
                if not journal or not matches:
                    continue
                if published and published < date.today() - timedelta(days=LOOKBACK_DAYS):
                    continue
                external = item.get("externalIds") or {}
                papers.append({
                    "title": title, "journal": journal,
                    "doi": (external.get("DOI") or "").lower(),
                    "url": item.get("url") or "", "abstract": abstract,
                    "published": published,
                    "authors": ", ".join(a.get("name", "") for a in item.get("authors", [])),
                    "keywords": matches, "sources": ["Semantic Scholar"],
                    "citation_count": item.get("citationCount") or 0,
                })
        except Exception as exc:
            print(f"Semantic Scholar warning for {keyword}: {exc}")
        time.sleep(1.1 if not SEMANTIC_SCHOLAR_API_KEY else 0.1)
    return papers


def xml_name(tag):
    return tag.rsplit("}", 1)[-1].lower()


def xml_value(node, *names):
    wanted = {name.lower() for name in names}
    for child in node.iter():
        if xml_name(child.tag) in wanted:
            if xml_name(child.tag) == "link" and child.get("href"):
                return child.get("href")
            if child.text:
                return child.text
    return ""


def rss_entries(content):
    root = ET.fromstring(content)
    nodes = [node for node in root.iter() if xml_name(node.tag) in {"item", "entry"}]
    for node in nodes:
        yield {
            "title": xml_value(node, "title"),
            "summary": xml_value(node, "summary", "description", "content", "encoded"),
            "link": xml_value(node, "link"),
            "id": xml_value(node, "id", "guid", "identifier"),
            "doi": xml_value(node, "doi"),
            "author": xml_value(node, "author", "creator", "name"),
            "date": xml_value(node, "published", "updated", "pubDate", "date"),
        }


def rss_date(value):
    parsed = parse_date(value)
    if parsed:
        return parsed
    try:
        from email.utils import parsedate_to_datetime
        return parsedate_to_datetime(value).date()
    except (TypeError, ValueError):
        return None


def extract_doi(entry):
    candidates = " ".join([entry.get("id", ""), entry.get("link", ""), entry.get("doi", "")])
    match = re.search(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", candidates, re.I)
    return match.group(0).rstrip(".,;)").lower() if match else ""


def fetch_rss():
    print("Fetching publisher RSS feeds...")
    papers = []
    for source, feeds in RSS_FEEDS.items():
        for journal, url in feeds:
            try:
                response = SESSION.get(
                    url, timeout=30,
                    headers={"Accept": "application/rss+xml, application/xml, text/xml"},
                )
                response.raise_for_status()
                for entry in rss_entries(response.content):
                    title = clean_text(entry.get("title", ""))
                    abstract = clean_text(entry.get("summary", "") or entry.get("description", ""))
                    matches = matched_keywords(title, abstract)
                    published = rss_date(entry.get("date"))
                    if not matches:
                        continue
                    if published and published < date.today() - timedelta(days=LOOKBACK_DAYS):
                        continue
                    papers.append({
                        "title": title, "journal": journal,
                        "doi": extract_doi(entry), "url": entry.get("link", ""),
                        "abstract": abstract, "published": published,
                        "authors": clean_text(entry.get("author", "")),
                        "keywords": matches, "sources": [source],
                        "citation_count": 0,
                    })
            except Exception as exc:
                print(f"{source} warning for {journal}: {exc}")
    return papers


def paper_key(paper):
    return f"doi:{paper['doi']}" if paper.get("doi") else f"title:{normalize(paper['title'])}"


def merge_papers(papers):
    merged = {}
    for paper in papers:
        key = paper_key(paper)
        if key not in merged:
            merged[key] = paper
            continue
        current = merged[key]
        current["sources"] = sorted(set(current["sources"] + paper["sources"]))
        current["keywords"] = sorted(set(current["keywords"] + paper["keywords"]))
        current["citation_count"] = max(current.get("citation_count", 0), paper.get("citation_count", 0))
        for field in ("abstract", "authors", "url", "doi", "published"):
            if not current.get(field) and paper.get(field):
                current[field] = paper[field]
    result = list(merged.values())
    result.sort(key=lambda p: (
        p.get("published") or date.min,
        len(p.get("keywords", [])), p.get("citation_count", 0),
    ), reverse=True)
    return result[:MAX_RESULTS]


def fallback_analysis(paper):
    keyword_text = "、".join(paper["keywords"][:3]) or "柔性传感"
    return {
        "innovation": f"文章围绕{keyword_text}展开；需查看原文确认具体创新。",
        "relevance": min(5, max(1, 2 + len(paper["keywords"]))),
        "inspiration": f"可重点参考其{keyword_text}相关材料体系、器件结构与测试方法。",
    }


def analyze_paper(paper):
    fallback = fallback_analysis(paper)
    if not DEEPSEEK_API_KEY:
        return fallback
    prompt = f"""你是柔性多维力传感器研究助手。只根据给定题目和摘要，输出严格 JSON：
{{"innovation":"一句中文，说明核心创新","relevance":1到5的整数,"inspiration":"一句中文，说明对离子凝胶、离子电容或三维力传感课题可借鉴之处"}}
期刊：{paper['journal']}
题目：{paper['title']}
命中关键词：{', '.join(paper['keywords'])}
摘要：{paper['abstract'] or '未提供摘要，请依据题目保守判断'}"""
    try:
        client = OpenAI(base_url=DEEPSEEK_BASE_URL, api_key=DEEPSEEK_API_KEY)
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}, temperature=0.2,
        )
        data = json.loads(response.choices[0].message.content)
        return {
            "innovation": clean_text(str(data.get("innovation", fallback["innovation"]))),
            "relevance": min(5, max(1, int(data.get("relevance", fallback["relevance"])))),
            "inspiration": clean_text(str(data.get("inspiration", fallback["inspiration"]))),
        }
    except Exception as exc:
        print(f"AI analysis warning for {paper['title']}: {exc}")
        return fallback


def generate_email(papers):
    today = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d")
    cards = []
    for paper in papers:
        analysis = paper["analysis"]
        stars = "★" * analysis["relevance"] + "☆" * (5 - analysis["relevance"])
        target = paper.get("url") or (f"https://doi.org/{paper['doi']}" if paper.get("doi") else "#")
        link = html.escape(target)
        published = paper["published"].isoformat() if paper.get("published") else "日期待确认"
        cards.append(f"""
        <article style="background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:22px;margin:18px 0;">
          <div style="font-size:18px;font-weight:700;color:#7c3aed;">⭐ {html.escape(paper['journal'])}</div>
          <div style="margin-top:12px;color:#6b7280;font-size:13px;">{published} · {html.escape(' / '.join(paper['sources']))}</div>
          <h2 style="font-size:20px;line-height:1.45;margin:8px 0 16px;"><a href="{link}" style="color:#111827;text-decoration:none;">{html.escape(paper['title'])}</a></h2>
          <div style="margin:10px 0;"><b>创新：</b>{html.escape(analysis['innovation'])}</div>
          <div style="margin:10px 0;"><b>与你课题相关：</b><span style="color:#f59e0b;font-size:20px;letter-spacing:2px;">{stars}</span></div>
          <div style="margin:10px 0;"><b>可借鉴：</b>{html.escape(analysis['inspiration'])}</div>
          <div style="margin-top:14px;color:#6b7280;font-size:13px;">关键词：{html.escape('、'.join(paper['keywords']))}</div>
        </article>""")
    if not cards:
        cards.append("<div style='background:#fff;padding:24px;border-radius:12px;'>过去 7 天内，目标期刊中没有发现命中关键词的新论文。</div>")
    return f"""
    <html><body style="margin:0;background:#f3f4f6;color:#1f2937;font-family:Arial,'Microsoft YaHei',sans-serif;">
      <main style="max-width:760px;margin:auto;padding:24px;">
        <header style="background:linear-gradient(135deg,#6d28d9,#2563eb);color:white;padding:28px;border-radius:14px;">
          <h1 style="margin:0;font-size:27px;">{today} 柔性传感日报</h1>
          <p style="margin:10px 0 0;opacity:.9;">目标期刊 · 关键词过滤 · 多数据源去重</p>
        </header>
        {''.join(cards)}
        <footer style="text-align:center;color:#9ca3af;font-size:12px;padding:18px;">Crossref · Semantic Scholar · Nature RSS · ACS RSS · Wiley RSS</footer>
      </main>
    </body></html>"""


def send_email(subject, content):
    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"] = SENDER_EMAIL
    message["To"] = RECEIVER_EMAIL
    message.attach(MIMEText(content, "html", "utf-8"))
    smtp_class = smtplib.SMTP_SSL if SMTP_PORT == 465 else smtplib.SMTP
    with smtp_class(SMTP_SERVER, SMTP_PORT, timeout=60) as server:
        if SMTP_PORT != 465:
            server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(message)


def main():
    missing = [name for name in ("SENDER_EMAIL", "SENDER_PASSWORD", "RECEIVER_EMAIL") if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"Missing required configuration: {', '.join(missing)}")
    papers = merge_papers(fetch_crossref() + fetch_semantic_scholar() + fetch_rss())
    print(f"Selected {len(papers)} journal papers after whitelist, keyword filtering, and deduplication")
    for paper in papers:
        print(f"- {paper['journal']}: {paper['title']}")
        paper["analysis"] = analyze_paper(paper)
    today = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d")
    send_email(f"{today} 柔性传感日报", generate_email(papers))
    print("Email sent successfully")


if __name__ == "__main__":
    main()
