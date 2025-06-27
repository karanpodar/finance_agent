from markitdown import MarkItDown
from openai import OpenAI
import requests
import re
import yfinance as yf
import pandas as pd
import os
import time
import random
from typing import List, Dict, Optional
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StockNewsAnalyzer:
    def __init__(self):
        """Initialize the StockNewsAnalyzer with session and MarkItDown."""
        self.session = self._create_robust_session()
        self.md = MarkItDown(requests_session=self.session)
    
    def _create_robust_session(self) -> requests.Session:
        """Create a robust session with retry strategy and proper headers."""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            backoff_factor=1,
            raise_on_status=False
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set headers
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        return session
    
    def get_news(self, stock: str) -> Optional[List[Dict]]:
        """
        Fetch relevant news articles for a given stock ticker.

        Parameters:
        - stock (str): The stock ticker symbol.

        Returns:
        - List[Dict]: A list of dictionaries containing title, summary, URL, and publication date of relevant news articles.
        """
        try:
            logger.info(f"Fetching news for stock: {stock}")
            
            # Fetch the ticker object and retrieve its news
            ticker = yf.Ticker(stock)
            news = ticker.news

            if not news:
                logger.warning(f"No news found for {stock}")
                return []

            # Filter news with contentType='STORY'
            relevant_news = [
                item for item in news if item.get('content', {}).get('contentType') == 'STORY'
            ]

            if not relevant_news:
                logger.warning(f"No relevant news stories found for {stock}")
                return []

            all_news = []
            for i, item in enumerate(relevant_news[:10]):  # Limit to the first 10 news items
                try:
                    content = item.get('content', {})
                    url = content.get('canonicalUrl', {}).get('url')
                    
                    # Skip if URL is missing
                    if not url:
                        logger.warning(f"No URL found for news item {i}")
                        continue
                    
                    current_news = {
                        'title': content.get('title', 'No Title'),
                        'summary': content.get('summary', 'No Summary'),
                        'url': url,
                        'pubdate': content.get('pubDate', '').split('T')[0] if content.get('pubDate') else 'No Date',
                    }
                    all_news.append(current_news)
                    
                except Exception as e:
                    logger.error(f"Error processing news item {i}: {e}")
                    continue

            logger.info(f"Successfully fetched {len(all_news)} news articles for {stock}")
            return all_news

        except Exception as e:
            logger.error(f"An error occurred while fetching news for {stock}: {e}")
            return None

    def remove_links(self, text: str) -> str:
        """Clean unnecessary links and special characters from text."""
        if not text:
            return ""
        
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'www\.\S+', '', text)  # Remove www URLs
        text = re.sub(r'\[.*?\]', '', text)  # Remove markdown-style links
        text = re.sub(r'[#*()+\-]', '', text)  # Remove special characters (but keep newlines)
        text = re.sub(r'/\S*', '', text)  # Remove slashes
        text = re.sub(r'  +', ' ', text)  # Remove multiple spaces
        text = text.strip()
        return text

    def extract_news_with_retry(self, url: str, max_retries: int = 3) -> Optional[str]:
        """
        Extract news content from a URL with retry mechanism for 429 and 403 errors.
        
        Parameters:
        - url (str): The URL to extract content from
        - max_retries (int): Maximum number of retry attempts
        
        Returns:
        - Optional[str]: Extracted content or None if failed
        """
        for attempt in range(max_retries):
            try:
                logger.info(f"Extracting content from URL (attempt {attempt + 1}/{max_retries}): {url}")
                
                # Use MarkItDown to extract the content
                information_to_extract = self.md.convert(url)
                
                if not information_to_extract:
                    logger.warning(f"No content extracted from {url}")
                    return None
                
                text_title = getattr(information_to_extract, 'title', '') or ''
                text_content = getattr(information_to_extract, 'text_content', '') or ''
                
                # Clean and combine the title and content
                full_content = f"{text_title.strip()}\n{text_content.strip()}"
                cleaned_content = self.remove_links(full_content)
                
                if cleaned_content:
                    logger.info(f"Successfully extracted content from {url}")
                    return cleaned_content
                else:
                    logger.warning(f"Extracted content is empty for {url}")
                    return None
                    
            except requests.exceptions.HTTPError as e:
                if e.response.status_code in [403, 429]:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)  # Exponential backoff
                    logger.warning(f"HTTP {e.response.status_code} error for {url}. Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"HTTP error {e.response.status_code} for {url}: {e}")
                    break
            except Exception as e:
                logger.error(f"Error extracting content from {url} (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    wait_time = random.uniform(1, 3)
                    logger.info(f"Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                else:
                    break
        
        logger.error(f"Failed to extract content from {url} after {max_retries} attempts")
        return None

    def extract_full_news(self, stock: str) -> List[Dict]:
        """
        Fetch full news articles with extracted content.

        Parameters:
        - stock (str): The stock ticker symbol.

        Returns:
        - List[Dict]: A list of dictionaries containing full_news of relevant news articles.
        """
        logger.info(f"Starting full news extraction for stock: {stock}")
        
        # Step 1: Fetch news using the get_news function
        news = self.get_news(stock)
        
        if not news:
            logger.warning(f"No news articles found for {stock}")
            return []

        # Step 2: Iterate through each news article
        successful_extractions = 0
        for i, item in enumerate(news):
            try:
                # Step 3: Extract the full news content using the URL
                full_news = self.extract_news_with_retry(item['url'])
                
                if full_news:
                    item['full_news'] = full_news
                    successful_extractions += 1
                else:
                    item['full_news'] = f"Failed to extract content. Summary: {item.get('summary', 'No summary available')}"
                
                # Add delay between requests to be respectful
                time.sleep(random.uniform(2, 5))
                
            except Exception as e:
                logger.error(f"Error processing news item {i}: {e}")
                item['full_news'] = f"Error extracting content: {str(e)}"
                continue

        logger.info(f"Successfully extracted content for {successful_extractions}/{len(news)} articles")
        return news

    def analyze_news(self, stock: str, articles: List[Dict]) -> Optional[str]:
        """
        Analyze news articles using LLM and provide sentiment analysis and investment recommendation.
        
        Parameters:
        - stock (str): The stock ticker symbol
        - articles (List[Dict]): List of news articles
        
        Returns:
        - Optional[str]: Analysis result or None if failed
        """
        try:
            # Check if OpenAI API key is available
            api_key = os.getenv("OpenAI_Alfred_API_KEY")
            if not api_key:
                logger.error("OpenAI API key not found in environment variables")
                return None
            
            # Step 1: Initialize the LLM client
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key
            )

            # Step 2: Prepare articles text
            articles_text = ""
            for i, article in enumerate(articles, 1):
                articles_text += f"\n\nArticle {i}:\n"
                articles_text += f"Title: {article.get('title', 'No Title')}\n"
                articles_text += f"Date: {article.get('pubdate', 'No Date')}\n"
                articles_text += f"Summary: {article.get('summary', 'No Summary')}\n"
                if article.get('full_news'):
                    # Limit content length to avoid token limits
                    full_news = article['full_news'][:2000] + "..." if len(article['full_news']) > 2000 else article['full_news']
                    articles_text += f"Full Content: {full_news}\n"

            logger.info(f"Artciles text: {articles_text}")

            # Step 3: Define the prompt template
            prompt = f"""
You are an expert financial analyst. I will provide you with news articles related to {stock}. Your tasks are:

1. **Sentiment Analysis:**
- For each news article, evaluate its sentiment as 'Positive', 'Negative', or 'Neutral'.
- Present your evaluation in a JSON dictionary format.

2. **Comprehensive Summary and Investment Recommendation:**
- Provide a concise summary of the overall sentiment and key points.
- Advise whether investing in {stock} is advisable, with reasons from the news analysis.

**News Articles:**
{articles_text}

**Output Format:**

1. **Sentiment Analysis Dictionary:**
```json
{{
    "Article Title 1": "Positive",
    "Article Title 2": "Negative",
    "Article Title 3": "Neutral"
}}
```

2. **Summary:** [Your summary here]

3. **Investment Recommendation:** [Your recommendation here]
"""

            # Step 4: Make API call with retry mechanism
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = client.chat.completions.create(
                        extra_headers={
                            "HTTP-Referer": "https://karan-alfredchatbot.streamlit.app/",
                            "X-Title": "Alfred Chatbot",
                        },
                        model="deepseek/deepseek-r1-0528-qwen3-8b:free",
                        messages=[
                            {"role": "system", "content": "You are an expert financial analyst specializing in stock market sentiment analysis."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.2,
                        max_tokens=2000
                    )
                    
                    analysis_result = response.choices[0].message.content
                    logger.info("Successfully received analysis from LLM")
                    return analysis_result
                    
                except Exception as e:
                    logger.error(f"API call attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                        logger.info(f"Retrying in {wait_time:.2f} seconds...")
                        time.sleep(wait_time)
                    else:
                        logger.error("All API call attempts failed")
                        return None

        except Exception as e:
            logger.error(f"Error in analyze_news: {e}")
            return None

    def generate_html_report(self, results: Dict) -> str:
        """
        Generate an HTML report from the analysis results.
        
        Parameters:
        - results (Dict): Analysis results
        
        Returns:
        - str: HTML content
        """
        from datetime import datetime
        
        # Extract sentiment analysis from the LLM response if available
        sentiment_json = ""
        summary = ""
        recommendation = ""
        
        if results.get('analysis'):
            analysis_text = results['analysis']
            
            # Try to extract JSON sentiment analysis
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', analysis_text, re.DOTALL)
            if json_match:
                try:
                    sentiment_data = json.loads(json_match.group(1))
                    sentiment_json = json.dumps(sentiment_data, indent=2)
                except Exception:
                    sentiment_json = json_match.group(1)
            else:
                # Fallback: Look for the first '{' and the first '}' after that
                start = analysis_text.find('{')
                end = analysis_text.find('}', start)
                if start != -1 and end != -1:
                    raw_json = analysis_text[start:end+1]
                    try:
                        sentiment_data = json.loads(raw_json)
                        sentiment_json = json.dumps(sentiment_data, indent=2)
                    except Exception:
                        sentiment_json = raw_json
            
            # Extract summary and recommendation
            summary_match = re.search(
                r'\*\*\s*Summary\s*:\s*\*\*\s*(.+?)(?=\*\*\s*Investment Recommendation\s*:\s*\*\*|$)',
                analysis_text,
                re.DOTALL | re.IGNORECASE
            )
            if summary_match:
                summary = summary_match.group(1).strip()
            else:
                # Fallback: find text after any form of "summary" up to "investment recommendation"
                fallback_summary = re.search(
                    r'Summary\s*:?\s*(.+?)(?=Investment Recommendation\s*:?|$)',
                    analysis_text,
                    re.DOTALL | re.IGNORECASE
                )
                if fallback_summary:
                    summary = fallback_summary.group(1).strip()

            # Primary pattern for investment recommendation
            recommendation_match = re.search(
                r'\*\*\s*Investment Recommendation\s*:\s*\*\*\s*(.+)',
                analysis_text,
                re.DOTALL | re.IGNORECASE
            )
            if recommendation_match:
                recommendation = recommendation_match.group(1).strip()
            else:
                # Fallback: get everything after "Investment Recommendation"
                fallback_reco = re.search(
                    r'Investment Recommendation\s*:?\s*(.+)',
                    analysis_text,
                    re.DOTALL | re.IGNORECASE
                )
                if fallback_reco:
                    recommendation = fallback_reco.group(1).strip()

        
        # Generate articles table
        articles_html = ""
        if results.get('news_articles'):
            for i, article in enumerate(results['news_articles'], 1):
                articles_html += f"""
                <tr>
                    <td>{i}</td>
                    <td><strong>{article.get('title', 'No Title')}</strong></td>
                    <td>{article.get('pubdate', 'No Date')}</td>
                    <td>{article.get('summary', 'No Summary')[:200]}{'...' if len(article.get('summary', '')) > 200 else ''}</td>
                    <td><a href="{article.get('url', '#')}" target="_blank">View</a></td>
                </tr>
                """
        
        # Generate sentiment table
        sentiment_html = ""
        if sentiment_json:
            try:
                import json
                sentiment_data = json.loads(sentiment_json) if isinstance(sentiment_json, str) else sentiment_json
                for title, sentiment in sentiment_data.items():
                    color_class = {
                        'Positive': 'positive',
                        'Negative': 'negative',
                        'Neutral': 'neutral'
                    }.get(sentiment, 'neutral')
                    
                    sentiment_html += f"""
                    <tr>
                        <td>{title[:80]}{'...' if len(title) > 80 else ''}</td>
                        <td><span class="sentiment {color_class}">{sentiment}</span></td>
                    </tr>
                    """
            except:
                sentiment_html = f"<tr><td colspan='2'>Error parsing sentiment data</td></tr>"
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stock Analysis Report - {results.get('stock', 'Unknown')}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .header {{
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            text-align: center;
        }}
        
        .header h1 {{
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 700;
        }}
        
        .header p {{
            color: #7f8c8d;
            font-size: 1.1em;
        }}
        
        .card {{
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        
        .card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2);
        }}
        
        .card h2 {{
            color: #2c3e50;
            margin-bottom: 20px;
            font-size: 1.8em;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        
        .status {{
            display: inline-block;
            padding: 8px 16px;
            border-radius: 25px;
            font-weight: bold;
            text-transform: uppercase;
            font-size: 0.9em;
            margin-bottom: 20px;
        }}
        
        .status.success {{
            background: linear-gradient(135deg, #2ecc71, #27ae60);
            color: white;
        }}
        
        .status.error {{
            background: linear-gradient(135deg, #e74c3c, #c0392b);
            color: white;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }}
        
        th, td {{
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid #ecf0f1;
        }}
        
        th {{
            background: linear-gradient(135deg, #3498db, #2980b9);
            color: white;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.9em;
            letter-spacing: 0.5px;
        }}
        
        tr:hover {{
            background-color: #f8f9fa;
        }}
        
        .sentiment {{
            padding: 5px 12px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.85em;
        }}
        
        .sentiment.positive {{
            background: linear-gradient(135deg, #2ecc71, #27ae60);
            color: white;
        }}
        
        .sentiment.negative {{
            background: linear-gradient(135deg, #e74c3c, #c0392b);
            color: white;
        }}
        
        .sentiment.neutral {{
            background: linear-gradient(135deg, #95a5a6, #7f8c8d);
            color: white;
        }}
        
        .summary-box {{
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            border-left: 5px solid #3498db;
            padding: 20px;
            margin: 15px 0;
            border-radius: 0 10px 10px 0;
            font-size: 1.05em;
            line-height: 1.7;
        }}
        
        .recommendation-box {{
            background: linear-gradient(135deg, #e8f5e8, #d4edda);
            border-left: 5px solid #28a745;
            padding: 20px;
            margin: 15px 0;
            border-radius: 0 10px 10px 0;
            font-size: 1.05em;
            line-height: 1.7;
            font-weight: 500;
        }}
        
        .error-box {{
            background: linear-gradient(135deg, #fdf2f2, #fed7d7);
            border-left: 5px solid #e53e3e;
            padding: 20px;
            margin: 15px 0;
            border-radius: 0 10px 10px 0;
            color: #c53030;
        }}
        
        a {{
            color: #3498db;
            text-decoration: none;
            font-weight: 600;
            transition: color 0.3s ease;
        }}
        
        a:hover {{
            color: #2980b9;
            text-decoration: underline;
        }}
        
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: rgba(255, 255, 255, 0.8);
            font-size: 0.9em;
        }}
        
        @media (max-width: 768px) {{
            .container {{
                padding: 10px;
            }}
            
            .header h1 {{
                font-size: 2em;
            }}
            
            .card {{
                padding: 20px;
            }}
            
            table {{
                font-size: 0.9em;
            }}
            
            th, td {{
                padding: 10px 8px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📈 Stock Analysis Report</h1>
            <p>Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        </div>
        
        <div class="card">
            <h2>📊 Analysis Summary</h2>
            <div class="status {'success' if results.get('success') else 'error'}">
                {'Analysis Completed Successfully' if results.get('success') else 'Analysis Failed'}
            </div>
            
            <p><strong>Stock Symbol:</strong> {results.get('stock', 'Unknown')}</p>
            <p><strong>News Articles Found:</strong> {len(results.get('news_articles', []))}</p>
            <p><strong>Status:</strong> {'Success' if results.get('success') else 'Failed'}</p>
            
            {f'<div class="error-box"><strong>Error:</strong> {results.get("error")}</div>' if results.get('error') else ''}
        </div>
        
        {f'''
        <div class="card">
            <h2>📰 News Articles</h2>
            <table>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Title</th>
                        <th>Date</th>
                        <th>Summary</th>
                        <th>Link</th>
                    </tr>
                </thead>
                <tbody>
                    {articles_html}
                </tbody>
            </table>
        </div>
        ''' if results.get('news_articles') else ''}
        
        {f'''
        <div class="card">
            <h2>💭 Sentiment Analysis</h2>
            <table>
                <thead>
                    <tr>
                        <th>Article Title</th>
                        <th>Sentiment</th>
                    </tr>
                </thead>
                <tbody>
                    {sentiment_html}
                </tbody>
            </table>
        </div>
        ''' if sentiment_html else ''}
        
        {f'''
        <div class="card">
            <h2>📝 Summary</h2>
            <div class="summary-box">
                {summary if summary else 'No summary available'}
            </div>
        </div>
        ''' if results.get('analysis') else ''}
        
        {f'''
        <div class="card">
            <h2>💡 Investment Recommendation</h2>
            <div class="recommendation-box">
                {recommendation if recommendation else 'No recommendation available'}
            </div>
        </div>
        ''' if results.get('analysis') else ''}
        
        {f'''
        <div class="card">
            <h2>🔍 Full Analysis</h2>
            <div class="summary-box">
                <pre style="white-space: pre-wrap; font-family: inherit;">{results.get('analysis', 'No analysis available')}</pre>
            </div>
        </div>
        ''' if results.get('analysis') else ''}
        
        <div class="footer">
            <p>📊 Stock News Analysis Tool | Generated using AI-powered sentiment analysis</p>
        </div>
    </div>
</body>
</html>
        """
        
        return html_content

    def save_html_report(self, results: Dict, filename: str = None) -> str:
        """
        Save the analysis results as an HTML file.
        
        Parameters:
        - results (Dict): Analysis results
        - filename (str): Optional filename, auto-generated if not provided
        
        Returns:
        - str: Path to the saved HTML file
        """
        if not filename:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            stock = results.get('stock', 'UNKNOWN')
            filename = f"reports/stock_analysis_{stock}_{timestamp}.html"
        
        html_content = self.generate_html_report(results)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"HTML report saved to: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error saving HTML report: {e}")
            raise

    def run_analysis(self, stock: str, save_html: bool = True, html_filename: str = None) -> Dict:
        """
        Run complete analysis for a stock symbol.
        
        Parameters:
        - stock (str): The stock ticker symbol
        - save_html (bool): Whether to save HTML report
        - html_filename (str): Optional HTML filename
        
        Returns:
        - Dict: Analysis results including HTML file path
        """
        results = {
            'stock': stock,
            'news_articles': [],
            'analysis': None,
            'success': False,
            'html_file': None
        }
        
        try:
            # Step 1: Get basic news articles
            logger.info(f"Starting analysis for stock: {stock}")
            news_articles = self.get_news(stock)
            
            if not news_articles:
                results['error'] = "No news articles found"
                if save_html:
                    results['html_file'] = self.save_html_report(results, html_filename)
                return results
            
            results['news_articles'] = news_articles
            logger.info(f"Found {len(news_articles)} news articles")
            
            # Step 2: Extract full content
            full_news_articles = self.extract_full_news(stock)
            results['news_articles'] = full_news_articles
            
            # Step 3: Analyze news
            analysis = self.analyze_news(stock, full_news_articles)
            if analysis:
                results['analysis'] = analysis
                results['success'] = True
                logger.info("Analysis completed successfully")
            else:
                results['error'] = "Failed to analyze news articles"
                logger.error("Analysis failed")
            
            # Step 4: Save HTML report
            if save_html:
                try:
                    html_file = self.save_html_report(results, html_filename)
                    results['html_file'] = html_file
                except Exception as e:
                    logger.error(f"Failed to save HTML report: {e}")
                    results['html_save_error'] = str(e)
            
        except Exception as e:
            logger.error(f"Error in run_analysis: {e}")
            results['error'] = str(e)
            if save_html:
                try:
                    results['html_file'] = self.save_html_report(results, html_filename)
                except Exception as html_e:
                    logger.error(f"Failed to save error HTML report: {html_e}")
        
        return results


def main(ticker: str = None):
    """Main function to run the stock news analyzer."""
    analyzer = StockNewsAnalyzer()
    
    try:
        if ticker:
            stock = ticker.strip().upper()
            print(f"Using provided ticker: {stock}")
        else:
            stock = input("Enter the stock ticker symbol for sentiment analysis: ").strip().upper()
        
        if not stock:
            stock = "AAPL"  # Default to Apple Inc.
            print(f"Using default stock: {stock}")
        
        # Ask user if they want to save HTML report
        save_html = input("Save HTML report? (y/n, default: y): ").strip().lower()
        save_html = save_html != 'n'
        
        html_filename = None
        if save_html:
            custom_filename = input("Enter custom filename (or press Enter for auto-generated): ").strip()
            if custom_filename:
                if not custom_filename.endswith('.html'):
                    custom_filename += '.html'
                html_filename = custom_filename
        
        # Run the complete analysis
        results = analyzer.run_analysis(stock, save_html=save_html, html_filename=html_filename)
        
        if results['success']:
            print(f"\n{'='*60}")
            print(f"ANALYSIS RESULTS FOR {results['stock']}")
            print(f"{'='*60}")
            
            # Display news articles summary
            print(f"\nFound {len(results['news_articles'])} news articles:")
            for i, article in enumerate(results['news_articles'], 1):
                print(f"{i}. {article['title']} ({article['pubdate']})")
            
            # Display analysis
            print(f"\n{'='*60}")
            print("SENTIMENT ANALYSIS & INVESTMENT RECOMMENDATION")
            print(f"{'='*60}")
            print(results['analysis'])
            
            # HTML report info
            if results.get('html_file'):
                print(f"\n{'='*60}")
                print("HTML REPORT SAVED")
                print(f"{'='*60}")
                print(f"📄 Report saved to: {results['html_file']}")
                print(f"🌐 Open in browser to view the formatted report")
            
        else:
            print(f"\nAnalysis failed for {results['stock']}")
            if 'error' in results:
                print(f"Error: {results['error']}")
            
            # Still show basic news if available
            if results['news_articles']:
                print(f"\nHowever, found {len(results['news_articles'])} news articles:")
                df = pd.DataFrame(results['news_articles'])
                print(df[['title', 'pubdate', 'summary']])
            
            # HTML report info even for failed analysis
            if results.get('html_file'):
                print(f"\n📄 HTML report with available data saved to: {results['html_file']}")
        
        # Handle HTML save errors
        if results.get('html_save_error'):
            print(f"\n⚠️  Warning: Failed to save HTML report - {results['html_save_error']}")
                
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()