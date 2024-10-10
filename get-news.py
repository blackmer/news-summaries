import requests
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse
from fuzzywuzzy import fuzz  # Import fuzzy matching library
from openai import OpenAI  # Import the OpenAI library
from dotenv import load_dotenv
import os
import csv
import traceback  # To print detailed error information
import logging
import warnings

# Suppress the SequenceMatcher warning
warnings.filterwarnings("ignore", category=UserWarning, module='fuzzywuzzy')

load_dotenv()

# Load sensitive information from .env file (without default values)
newsapi_api_key = os.getenv("NEWSAPI_API_KEY")
diffbot_api_key = os.getenv("DIFFBOT_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
query = os.getenv("QUERY")  # No default value
exclude_domains = os.getenv("EXCLUDE_DOMAINS")  # This should remain as a string
competitors = os.getenv("COMPETITORS").split(',')  # No default value

# Parse DOMAIN_SCORES from .env (comma-separated list of domain:score pairs)
domain_scores = dict((item.split(":")[0], int(item.split(":")[1])) for item in os.getenv("DOMAIN_SCORES").split(",") if item)

# Optional: raise an error if any critical environment variables are missing
if not all([newsapi_api_key, diffbot_api_key, openai_api_key, query, exclude_domains, competitors, domain_scores]):
    raise ValueError("One or more required environment variables are missing.")


# Load the folder path from the .env file (without default value)
output_folder = os.getenv("OUTPUT_FOLDER")

# Raise an error if OUTPUT_FOLDER is not set in the .env file
if not output_folder:
    raise ValueError("The OUTPUT_FOLDER environment variable is missing. Please set it in the .env file.")


def contains_competitor(description, competitors):
    """Check if any competitor is mentioned in the description."""
    description = description.lower()  # Convert to lowercase for case-insensitive comparison
    for competitor in competitors:
        if competitor.lower() in description:
            return True
    return False

def get_domain_from_url(article_url):
    """Extract the domain from the article URL."""
    parsed_url = urlparse(article_url)
    domain = parsed_url.netloc
    return domain

def get_domain_score(article_url, domain_scores):
    """Return the score for the domain extracted from the article URL."""
    domain = get_domain_from_url(article_url)
    return domain_scores.get(domain, 0)

def is_russian_domain(domain):
    """Check if the domain ends with .ru."""
    return domain.endswith('.ru')

def get_article_text_from_diffbot(url):
    """Fetch the full article text using Diffbot."""
    logging.info(f"Fetching article text from Diffbot for URL: {url}")
    diffbot_url = f'https://api.diffbot.com/v3/article?token={diffbot_api_key}&url={url}'
    try:
        response = requests.get(diffbot_url, timeout=10)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        data = response.json()
        if 'objects' in data and len(data['objects']) > 0:
            article_text = data['objects'][0].get('text', 'No text available')
            logging.info(f"Successfully fetched article text (length: {len(article_text)} characters)")
            return article_text
        else:
            logging.warning(f"No content found for URL: {url}")
            return 'No content found'
    except requests.exceptions.Timeout:
        logging.error(f"Timeout error when fetching article from Diffbot: {url}")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching article from Diffbot: {e}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON response from Diffbot: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error in get_article_text_from_diffbot: {e}")
        return None

def send_to_gpt(article_text, openai_api_key):
    logging.info("Sending article to GPT for analysis")
    client = OpenAI(api_key=openai_api_key)
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{user_prompt}\n\nArticle Text:\n{article_text}\n\nPlease provide five blog post angles."}
            ],
            max_tokens=500
        )
        logging.info("Successfully received response from GPT")
        return response.choices[0].message.content.strip()
    except openai.APIError as e:
        logging.error(f"OpenAI API error: {e}")
        print(f"Error with OpenAI API: {e}")
        return "Error generating blog angles."
    except openai.RateLimitError as e:
        logging.error(f"OpenAI rate limit exceeded: {e}")
        print("Rate limit exceeded. Please try again later.")
        return "Error: Rate limit exceeded."
    except openai.AuthenticationError as e:
        logging.error(f"OpenAI authentication error: {e}")
        print("Authentication error. Please check your API key.")
        return "Error: Authentication failed."
    except Exception as e:
        logging.error(f"Unexpected error in send_to_gpt: {e}")
        print(f"An unexpected error occurred. Please check the log file for details.")
        return "Error generating blog angles."

# Set up logging to output to a file
logging.basicConfig(filename='script_output.log', level=logging.DEBUG, format='%(asctime)s - %(message)s')


def save_blog_angles_to_csv(title, source, published_at, description, article_url, blog_angles, folder_path=output_folder, filename=None):
    """
    Saves blog angles to a CSV file along with article details, ensuring proper alignment of columns
    and handling abstracts when provided. Now includes a timestamp in the filename.
    """

    try:
        # Print status message for the user and flush immediately
        print("\nGenerating titles and abstracts...", flush=True)
        
        logging.debug(f"save_blog_angles_to_csv called with: title={title}, source={source}, published_at={published_at}, description={description}, article_url={article_url}, blog_angles={blog_angles}")

        # Generate a timestamped filename if none is provided
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            filename = f"blog_angles_{timestamp}.csv"

        # If folder_path is provided, use it to construct the full file path
        if folder_path:
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)  # Create the folder if it doesn't exist
            filename = os.path.join(folder_path, filename)  # Update the filename to include the folder path

        with open(filename, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)

            # Write the headers if the file is empty
            if file.tell() == 0:
                writer.writerow(["Title", "Source", "Published At", "Description", "URL", "Blog Angle", "Abstract"])

            # Split the blog_angles string into separate angles (assuming each starts on a new line)
            angles = blog_angles.split("\n")
            logging.debug(f"Split angles: {angles}")

            # Initialize variables to keep track of blog angle and abstract together
            current_blog_angle = None
            current_abstract = None

            # Iterate over blog angles
            for angle in angles:
                angle = angle.strip()
                logging.debug(f"Processing angle: '{angle}'")

                if not angle:  # Skip empty lines
                    logging.debug("Skipping empty angle")
                    continue

                # Check if this line is an angle (begins with a number or some specific identifier, e.g., "1.", "2.", etc.)
                if angle.startswith(("1.", "2.", "3.", "4.", "5.")):
                    # If it's a new blog angle, write the previous angle (if any)
                    if current_blog_angle and current_abstract:
                        writer.writerow([title, source, published_at, description, article_url, current_blog_angle.strip(), current_abstract.strip()])
                        logging.debug(f"Wrote blog angle and abstract to CSV: '{current_blog_angle}' - '{current_abstract}'")

                    # Now set the new blog angle and clear the current abstract
                    current_blog_angle = angle
                    current_abstract = None  # Reset the abstract for the next block

                else:
                    # Concatenate abstract carefully without introducing any unintended characters
                    if current_blog_angle:
                        current_abstract = (current_abstract or "").strip() + " " + angle.strip()  # Add space between lines

            # Write the last blog angle and abstract, if any
            if current_blog_angle and current_abstract:
                writer.writerow([title, source, published_at, description, article_url, current_blog_angle.strip(), current_abstract.strip()])
                logging.debug(f"Wrote final blog angle and abstract to CSV: '{current_blog_angle}' - '{current_abstract}'")

        # Print completion message and flush immediately
        print("\nGeneration complete...", flush=True)

        logging.info(f"Blog angles and abstracts saved to {filename}")

    except Exception as e:
        logging.error(f"Error saving blog angles to CSV file: {e}")
        traceback.print_exc(file=open('script_output.log', 'a'))

def aggregate_articles_by_title(articles, threshold):
    """Aggregate similar articles based on title similarity."""
    aggregated = []
    while articles:
        base_article = articles.pop(0)
        similar_articles = [base_article]
        
        articles_to_keep = []
        for article in articles:
            if fuzz.ratio(base_article['title'], article['title']) >= threshold:
                similar_articles.append(article)
            else:
                articles_to_keep.append(article)
        articles = articles_to_keep
        aggregated.append(similar_articles)
    return aggregated

def fetch_news():
    # These could be moved outside the function if they're used elsewhere
    page_size = 10
    similarity_threshold = 70

    try:
        # Get the current time and X days ago in ISO 8601 format
        current_time = datetime.now(timezone.utc)  # Current UTC time
        time_x_days_ago = current_time - timedelta(days=5)

        # Convert to the format required by NewsAPI (YYYY-MM-DDTHH:MM:SS)
        from_date = time_x_days_ago.strftime('%Y-%m-%dT%H:%M:%SZ')
        to_date = current_time.strftime('%Y-%m-%dT%H:%M:%SZ')

        logging.info(f"Fetching news from {from_date} to {to_date}")

        # URL for fetching news articles
        url = (f'https://newsapi.org/v2/everything?qInTitle={query}&language=en&from={from_date}&to={to_date}'
               f'&excludeDomains={exclude_domains}&pageSize={page_size}&sortBy=publishedAt&apiKey={newsapi_api_key}')

        response = requests.get(url)
        if response.status_code == 200:
            articles = response.json().get('articles', [])
            logging.info(f"Found {len(articles)} articles")
            print(f"Found {len(articles)} articles published in the last 5 days.")

            filtered_articles = []
            
            # Filter out articles that mention competitors or come from .ru domains and calculate score
            for article in articles:
                description = article.get('description', 'No description available')
                article_url = article['url']
                domain = get_domain_from_url(article_url)
                
                if not contains_competitor(description, competitors) and not is_russian_domain(domain):
                    score = get_domain_score(article_url, domain_scores)
                    article['score'] = score
                    filtered_articles.append(article)

            # Aggregate similar articles based on title similarity
            aggregated_articles = aggregate_articles_by_title(filtered_articles, threshold=similarity_threshold)

            # Sort the aggregated articles by their highest score within each group
            sorted_aggregated_articles = sorted(aggregated_articles, key=lambda group: max([article['score'] for article in group]), reverse=True)

            # Print aggregated articles sorted by score
            print(f"Displaying {len(sorted_aggregated_articles)} aggregated topics, sorted by score:")
            for i, group in enumerate(sorted_aggregated_articles, 1):
                highest_scoring_article = max(group, key=lambda article: article['score'])
                
                print(f"\nTopic {i}: (Top score: {highest_scoring_article['score']})")
                
                for article in group:
                    title = article['title']
                    source = article['source']['name']
                    published_at = article['publishedAt']
                    description = article.get('description', 'No description available')
                    article_url = article['url']

                    # Display basic information for all articles
                    print(f"   Title: {title}")
                    print(f"   Source: {source}")
                    print(f"   Published At: {published_at}")
                    print(f"   Description: {description}")
                    print(f"   URL: {article_url}")
                    print(f"   Score: {article['score']}")
                    
                    # Fetch full text using Diffbot for the articles with score > 0
                    if article['score'] > 0:
                        full_text = get_article_text_from_diffbot(article_url)
                        print("\nFetching full article text...")

                        # Send the full text to GPT for analysis
                        blog_angle = send_to_gpt(full_text, openai_api_key)
                        print(f"\nArticle text being sent to the LLM for analysis...")
                        save_blog_angles_to_csv(title, source, published_at, description, article_url, blog_angle)

        else:
            logging.error(f"Error fetching news: HTTP {response.status_code}")
            print(f"Error fetching news: HTTP {response.status_code}")

    except requests.RequestException as e:
        logging.error(f"Error fetching news: {e}")
        print(f"Error fetching news: {e}")
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON response: {e}")
        print(f"Error parsing JSON response: {e}")
    except Exception as e:
        logging.error(f"Unexpected error in fetch_news: {e}")
        print(f"An unexpected error occurred. Please check the log file for details.")
        raise

# Note: This should be outside the fetch_news() function
if __name__ == "__main__":
    fetch_news()
