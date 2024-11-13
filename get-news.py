import asyncio
import aiohttp
from aiohttp import ClientTimeout
import requests
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse
from fuzzywuzzy import fuzz
from collections import defaultdict 
from openai import OpenAI
import yaml
from pathlib import Path
import os
import csv
import traceback
import logging
import warnings
import json
import re
from typing import Dict, Any, Optional

# Suppress the SequenceMatcher warning
warnings.filterwarnings("ignore", category=UserWarning, module='fuzzywuzzy')

# Set up logging
logging.basicConfig(filename='script_output.log', level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

class PerplexityAPI:
    """Class to handle Perplexity API interactions."""
    
    BASE_URL = "https://api.perplexity.ai/chat/completions"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("Perplexity API key not found in configuration")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    async def async_search_ddos_news(self) -> Optional[Dict[str, Any]]:
        """
        Asynchronously search for DDoS attack news using Perplexity's Sonar model.
        """
        payload = {
            "model": "llama-3.1-sonar-huge-128k-online",
            "messages": [{
                "role": "system",
                "content": "You are a news search assistant. Please search for and summarize recent news articles."
            }, {
                "role": "user",
                "content": "Search for news articles about DDoS attacks from the last 24 hours only. Provide only factual information from real news sources. For each article found, include the source, publication date, URL, and a brief summary."
            }]
        }
        
        try:
            logging.info("Making request to Perplexity API...")
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.BASE_URL,
                    headers=self.headers,
                    json=payload,
                    timeout=ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result
                    
                    error_text = await response.text()
                    logging.error(f"Perplexity API error: {error_text}")
                    return None
                    
        except Exception as e:
            logging.error(f"Error in Perplexity API request: {e}")
            return None

def parse_perplexity_results(results: Dict[str, Any]) -> list:
    """
    Parse the Perplexity API results into a format compatible with our existing article structure.
    """
    if not results or 'choices' not in results or not results['choices']:
        return []

    response_content = results['choices'][0].get('message', {}).get('content', '')
    if response_content.strip() == "NO_RECENT_ARTICLES":
        return []

    parsed_articles = []
    article_texts = response_content.split('\n\n')
    
    for article_text in article_texts:
        try:
            # Extract URL using regex
            url_match = re.search(r'https?://[^\s]+', article_text)
            url = url_match.group(0) if url_match else None
            
            if url:
                # Clean up title by removing Markdown formatting and number prefix
                title_line = article_text.split('\n')[0]
                title = re.sub(r'^\d+\.\s*', '', title_line)  # Remove number prefix
                title = re.sub(r'\*\*', '', title)  # Remove Markdown bold syntax
                
                article = {
                    'title': title,
                    'url': url,
                    'source': {'name': 'Perplexity API'},
                    'publishedAt': datetime.now().isoformat(),
                    'description': article_text,
                    'score': 1
                }
                parsed_articles.append(article)
        except Exception as e:
            logging.warning(f"Error parsing Perplexity article: {e}")
            continue
    
    return parsed_articles

def env_var_constructor(loader, node):
    value = loader.construct_scalar(node)
    env_var = os.getenv(value)
    if env_var is None:
        logging.warning(f"Environment variable {value} is not set")
        return f"${{{value}}}"  # Return the original placeholder if env var is not set
    return env_var

def load_config():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    # Function to replace environment variables
    def replace_env_vars(value):
        if isinstance(value, str):
            pattern = re.compile(r'\$\{([^}^{]+)\}')
            matches = pattern.findall(value)
            for match in matches:
                env_var = os.getenv(match)
                if env_var:
                    value = value.replace(f'${{{match}}}', env_var)
                else:
                    logging.warning(f"Environment variable {match} not set")
        return value

    # Process all values in the config
    for key in config:
        config[key] = replace_env_vars(config[key])
    
    # Log loaded configuration
    for key, value in config.items():
        if 'api_key' in key.lower():
            logging.info(f"Loaded config {key}: {'*' * 8}")  # Mask API keys in logs
        else:
            logging.info(f"Loaded config {key}: {value}")
    
    return config

def validate_config(config):
    required_keys = ['newsapi_api_key', 'diffbot_api_key', 'openai_api_key', 'perplexity_api_key', 'query', 'exclude_domains']
    for key in required_keys:
        if key not in config or not config[key]:
            raise ValueError(f"Missing or empty required configuration: {key}")
        if key.endswith('_api_key') and config[key].startswith('${'):
            raise ValueError(f"Environment variable for {key} not properly set")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Use it after loading the config
config = load_config()
validate_config(config)

# Validate critical configuration items
critical_items = ['newsapi_api_key', 'diffbot_api_key', 'openai_api_key', 'query', 'exclude_domains', 'competitors', 'domain_scores', 'output_folder']
for item in critical_items:
    if item not in config or not config[item]:
        raise ValueError(f"Critical configuration item '{item}' is missing or empty.")

def contains_competitor(description, competitors):
    """Check if any competitor is mentioned in the description."""
    if description is None:
        return False
    description = description.lower()  # Convert to lowercase for case-insensitive comparison
    for competitor in competitors:
        if competitor.lower() in description:
            return True
    return False

def get_domain_from_url(article_url):
    """Extract the domain from the article URL."""
    parsed_url = urlparse(article_url)
    domain = parsed_url.netloc
    # Remove 'www.' prefix if present
    if domain.startswith('www.'):
        domain = domain[4:]
    return domain

def get_domain_score(article_url, domain_scores):
    """Return the score for the domain extracted from the article URL."""
    domain = get_domain_from_url(article_url)
    score = domain_scores.get(domain, 0)
    logging.info(f"URL: {article_url}, Extracted domain: {domain}, Score: {score}")  # Add this line
    return score

def is_russian_domain(domain):
    """Check if the domain ends with .ru."""
    return domain.endswith('.ru')

async def fetch_article_text(session, url):
    diffbot_url = f'https://api.diffbot.com/v3/article?token={config["diffbot_api_key"]}&url={url}'
    try:
        async with session.get(diffbot_url, timeout=ClientTimeout(total=10)) as response:
            data = await response.json()
            if 'objects' in data and len(data['objects']) > 0:
                return data['objects'][0].get('text', 'No text available')
            return 'No content found'
    except Exception as e:
        logging.error(f"Error fetching article from Diffbot: {e}")
        return None
    
async def process_articles(articles):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_article_text(session, article['url']) for article in articles if article['score'] > 0]
        article_texts = await asyncio.gather(*tasks)
        
    for article, text in zip(articles, article_texts):
        if text:
            logging.info(f"Processing article: {article['title']}")
            print(f"\nFetching full article text for: {article['title']}")
            print(f"Article text being sent to the LLM for analysis...")
            blog_angle = send_to_gpt(text)
            logging.info(f"Blog angles generated for {article['title']}: {blog_angle[:100]}...")  # Log first 100 chars
            save_blog_angles_to_csv(
                article['title'], 
                article['source']['name'], 
                article['publishedAt'], 
                article.get('description', 'No description available'), 
                article['url'], 
                blog_angle
            )
            print(f"Blog angles generated and saved for: {article['title']}")
        else:
            logging.warning(f"No text retrieved for article: {article['title']}")

def send_to_gpt(article_text):
    logging.info("Sending article to GPT for analysis")
    client = OpenAI(api_key=config['openai_api_key'])
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": config['system_prompt']},
                {"role": "user", "content": f"{config['user_prompt']}\n\nArticle Text:\n{article_text[:500]}...\n\nPlease provide five blog post angles."}
            ],
            max_tokens=500
        )
        logging.info("Successfully received response from GPT")
        content = response.choices[0].message.content.strip()
        logging.debug(f"GPT response (first 100 chars): {content[:100]}...")
        return content
    except Exception as e:
        logging.error(f"Error in send_to_gpt: {e}")
        print(f"Error generating blog angles: {e}")
        return "Error generating blog angles."

# Set up logging to output to a file
logging.basicConfig(filename='script_output.log', level=logging.DEBUG, format='%(asctime)s - %(message)s')


def save_blog_angles_to_csv(title, source, published_at, description, article_url, blog_angles, filename=None):
    try:
        print("\nSaving blog angles to CSV...", flush=True)
        
        logging.debug(f"save_blog_angles_to_csv called with: title={title}, source={source}, published_at={published_at}, description={description}, article_url={article_url}, blog_angles={blog_angles}")

        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            filename = f"blog_angles_{timestamp}.csv"

        folder_path = config['output_folder']
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        filename = os.path.join(folder_path, filename)

        with open(filename, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)

            # Write the headers if the file is empty
            if file.tell() == 0:
                writer.writerow(["Title", "Source", "Published At", "Description", "URL", "Blog Angle", "Abstract"])

            # Process and write blog angles
            angles = blog_angles.split("\n")
            current_blog_angle = None
            current_abstract = None

            for angle in angles:
                angle = angle.strip()
                if angle.startswith(("1.", "2.", "3.", "4.", "5.")):
                    if current_blog_angle and current_abstract:
                        writer.writerow([title, source, published_at, description, article_url, current_blog_angle.strip(), current_abstract.strip()])
                    current_blog_angle = angle
                    current_abstract = None
                elif current_blog_angle:
                    current_abstract = (current_abstract or "").strip() + " " + angle.strip()

            # Write the last blog angle and abstract, if any
            if current_blog_angle and current_abstract:
                writer.writerow([title, source, published_at, description, article_url, current_blog_angle.strip(), current_abstract.strip()])

        print(f"Blog angles saved to {filename}", flush=True)
        logging.info(f"Blog angles and abstracts saved to {filename}")

    except Exception as e:
        logging.error(f"Error saving blog angles to CSV file: {e}")
        traceback.print_exc(file=open('script_output.log', 'a'))

def aggregate_articles_by_title(articles, threshold, max_group_size=5):
    """
    Aggregate similar articles based on title similarity using a more efficient algorithm.
    
    :param articles: List of article dictionaries
    :param threshold: Similarity threshold (0-100)
    :param max_group_size: Maximum number of articles in a group (default: 5)
    :return: List of aggregated article groups
    """
    # Sort articles by title to improve clustering efficiency
    sorted_articles = sorted(articles, key=lambda x: x['title'])
    
    groups = defaultdict(list)
    for article in sorted_articles:
        added_to_group = False
        for group_key in groups:
            if fuzz.ratio(group_key, article['title']) >= threshold:
                if len(groups[group_key]) < max_group_size:
                    groups[group_key].append(article)
                    added_to_group = True
                break
        if not added_to_group:
            groups[article['title']].append(article)
    
    return list(groups.values())

async def fetch_news():
    logging.info("Starting fetch_news function")
    try:
        # Initialize both news sources
        newsapi_articles = []
        perplexity_articles = []

        # Fetch from NewsAPI
        current_time = datetime.now(timezone.utc)
        time_x_days_ago = current_time - timedelta(days=config['days_to_fetch'])
        from_date = time_x_days_ago.strftime('%Y-%m-%dT%H:%M:%SZ')
        to_date = current_time.strftime('%Y-%m-%dT%H:%M:%SZ')

        logging.info(f"Fetching news from {from_date} to {to_date}")

        url = (f'https://newsapi.org/v2/everything?qInTitle={config["query"]}&language=en&from={from_date}&to={to_date}'
               f'&excludeDomains={config["exclude_domains"]}&pageSize={config["page_size"]}&sortBy=publishedAt&apiKey={config["newsapi_api_key"]}')

        response = requests.get(url)
        if response.ok:
            newsapi_articles = response.json().get('articles', [])
            logging.info(f"Found {len(newsapi_articles)} articles from NewsAPI")
            print(f"Found {len(newsapi_articles)} articles from NewsAPI")

        # Fetch from Perplexity API
        perplexity = PerplexityAPI(config['perplexity_api_key'])
        perplexity_results = await perplexity.async_search_ddos_news()
        if perplexity_results:
            perplexity_articles = parse_perplexity_results(perplexity_results)
            logging.info(f"Found {len(perplexity_articles)} articles from Perplexity")
            print(f"Found {len(perplexity_articles)} articles from Perplexity")

        # Combine articles from both sources
        all_articles = newsapi_articles + perplexity_articles
        logging.info(f"Total articles found: {len(all_articles)}")
        print(f"Total articles found: {len(all_articles)}")

        filtered_articles = []
        for article in all_articles:
            description = article.get('description', '')
            article_url = article['url']
            domain = get_domain_from_url(article_url)
        
            if not contains_competitor(description, config['competitors']) and not is_russian_domain(domain):
                score = get_domain_score(article_url, config['domain_scores'])
                article['score'] = score
                filtered_articles.append(article)

        logging.info(f"Filtered down to {len(filtered_articles)} articles")

        # Print detailed information for each filtered article
        print("\nDetailed Article Information:")
        for i, article in enumerate(filtered_articles, 1):
            print(f"\nArticle {i}:")
            print(f"Title: {article['title']}")
            print(f"Source: {article['source']['name']}")
            print(f"Published At: {article['publishedAt']}")
            print(f"URL: {article['url']}")
            print(f"Description: {article.get('description', 'No description available')}")
            print(f"Score: {article['score']}")

        # Aggregate similar articles based on title similarity
        aggregated_articles = aggregate_articles_by_title(
            filtered_articles, 
            threshold=config['similarity_threshold'],
            max_group_size=config.get('max_group_size', 5)
        )

        # Sort the aggregated articles by their highest score within each group
        sorted_aggregated_articles = sorted(
            aggregated_articles, 
            key=lambda group: max([article['score'] for article in group]), 
            reverse=True
        )

        # Print aggregated articles sorted by score
        print(f"\nDisplaying {len(sorted_aggregated_articles)} aggregated topics, sorted by score:")
        for i, group in enumerate(sorted_aggregated_articles, 1):
            highest_scoring_article = max(group, key=lambda article: article['score'])
            print(f"\nTopic {i}: (Top score: {highest_scoring_article['score']})")
            
            for article in group:
                print(f"   Title: {article['title']}")
                print(f"   Source: {article['source']['name']}")
                print(f"   Published At: {article['publishedAt']}")
                print(f"   URL: {article['url']}")
                print(f"   Score: {article['score']}")
    
        # Process all filtered articles
        await process_articles(filtered_articles)

    except requests.RequestException as e:
        logging.error(f"Error fetching news: {e}")
        print(f"Error fetching news: {e}")
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON response: {e}")
        print(f"Error parsing JSON response: {e}")
    except Exception as e:
        logging.error(f"Unexpected error in fetch_news: {e}", exc_info=True)
        print(f"An unexpected error occurred. Please check the log file for details.")
        raise

# Note: This should be outside the fetch_news() function
if __name__ == "__main__":
    try:
        logging.info("Script started")
        asyncio.run(fetch_news())
        logging.info("Script completed successfully")
    except Exception as e:
        logging.critical(f"Critical error in main execution: {e}")
        print(f"An error occurred. Please check the log file for details.")
    finally:
        logging.info("Script execution ended")
