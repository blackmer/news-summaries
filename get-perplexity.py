import os
import requests
from datetime import datetime, timedelta
import json
from typing import Dict, Any, Optional

class PerplexityAPI:
    """Class to handle Perplexity API interactions."""
    
    BASE_URL = "https://api.perplexity.ai/chat/completions"
    
    def __init__(self):
        self.api_key = os.getenv('PERPLEXITY_API_KEY')
        if not self.api_key:
            raise ValueError("PERPLEXITY_API_KEY environment variable not found")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def search_ddos_news(self) -> Optional[Dict[str, Any]]:
        """
        Search for DDoS attack news from the last 24 hours using Perplexity's Sonar model.
        
        Returns:
            Dict containing the API response or None if the request fails
        """
        payload = {
            "model": "llama-3.1-sonar-huge-128k-online",
            "messages": [{
                "role": "system",
                "content": "You are a news search assistant. Please search for and summarize recent news articles."
            }, {
                "role": "user",
                "content": "Search for news articles about DDoS attacks from the last 24 hours only. Provide only factual information from real news sources. For each article found, include the source, publication date, URL, and a brief summary. If no articles from the last 24 hours are found, respond only with the exact phrase 'NO_RECENT_ARTICLES' and nothing else."
            }]
        }
        
        try:
            print("\nMaking request to Perplexity API...")
            print("Payload:", json.dumps(payload, indent=2))
            
            response = requests.post(
                self.BASE_URL,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            # Print response details for debugging
            print("\nResponse Status:", response.status_code)
            
            if response.ok:
                result = response.json()
                return result
            
            print("Error response:", response.text)
            response.raise_for_status()
            
        except requests.exceptions.RequestException as e:
            print(f"\nError making request to Perplexity API: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response status code: {e.response.status_code}")
                try:
                    print("Error response:", json.dumps(e.response.json(), indent=2))
                except json.JSONDecodeError:
                    print("Error response (raw):", e.response.text)
            return None

def main():
    """Main function to run the script."""
    try:
        perplexity = PerplexityAPI()
        results = perplexity.search_ddos_news()
        
        if results and 'choices' in results and results['choices']:
            response_content = results['choices'][0].get('message', {}).get('content', '')
            
            # Check if we have actual results
            if response_content.strip() == "NO_RECENT_ARTICLES":
                print("\nNo DDoS attack news articles found from the last 24 hours.")
                return
            
            print("\nRecent DDoS Attack News:")
            print(response_content)
            
            # Only save to file if we have actual recent results
            with open('ddos_news_results.txt', 'w') as f:
                f.write(response_content)
            print("\nResults have been saved to 'ddos_news_results.txt'")
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()