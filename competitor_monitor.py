from collections import defaultdict
from datetime import datetime
import csv
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

class CompetitorMonitor:
    def __init__(self, config: Dict):
        self.config = config
        self.competitor_mentions = defaultdict(list)
        self.setup_logging()
        logging.info("CompetitorMonitor initialized")
    
    def setup_logging(self):
        """Configure logging for the competitor monitor."""
        self.logger = logging.getLogger(__name__)
    
    def process_article(self, article: Dict) -> Tuple[str, List[str]]:
        """
        Process an article and track competitor mentions.
        Returns: Tuple of (article_type, list_of_mentioned_competitors)
        """
        description = article.get('description', '')
        mentioned_competitors = self.find_mentioned_competitors(description)
        
        if mentioned_competitors:
            article['competitors_mentioned'] = mentioned_competitors
            for competitor in mentioned_competitors:
                self.competitor_mentions[competitor].append({
                    'title': article.get('title', 'No title'),
                    'date': article.get('publishedAt', 'No date'),
                    'url': article.get('url', 'No URL'),
                    'context': self.extract_mention_context(description, competitor)
                })
            logging.info(f"Competitor mention found in article: {article.get('title', 'No title')}")
            return 'competitor', mentioned_competitors
        return 'normal', []
    
    def find_mentioned_competitors(self, text: Optional[str]) -> List[str]:
        """Find which competitors are mentioned in the text"""
        if not text:
            return []
        return [comp for comp in self.config['competitors'] 
                if comp.lower() in text.lower()]
    
    def extract_mention_context(self, text: str, competitor: str) -> str:
        """Extract the context around competitor mentions"""
        if not text:
            return ""
        
        # Convert both to lowercase for case-insensitive search
        text_lower = text.lower()
        competitor_lower = competitor.lower()
        
        # Find the competitor mention in the text
        index = text_lower.find(competitor_lower)
        if index == -1:
            return ""
            
        # Get 100 characters before and after, or as much as available
        start = max(0, index - 100)
        end = min(len(text), index + len(competitor) + 100)
        
        # Get the original case version of the context
        context = text[start:end].strip()
        
        # Add ellipsis if we truncated the text
        if start > 0:
            context = f"...{context}"
        if end < len(text):
            context = f"{context}..."
            
        return context
    
    def generate_competitor_report(self) -> Optional[str]:
        """Generate a report of competitor mentions."""
        if not self.competitor_mentions:
            logging.info("No competitor mentions to report")
            return None
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        output_path = Path('outputs') / f"competitor_mentions_{timestamp}.csv"
        output_path.parent.mkdir(exist_ok=True)
        
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Competitor', 'Article Title', 'Date', 'URL', 'Context'])
                
                for competitor, mentions in self.competitor_mentions.items():
                    for mention in mentions:
                        writer.writerow([
                            competitor,
                            mention['title'],
                            mention['date'],
                            mention['url'],
                            mention['context']
                        ])
            
            logging.info(f"Competitor report generated: {output_path}")
            return str(output_path)
        except Exception as e:
            logging.error(f"Error generating competitor report: {e}")
            return None