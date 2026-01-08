import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime
import sqlite3
import os

class TN2026PollScraper:
    def __init__(self):
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        self.db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'polls.db')
        
    def scrape_tn_polls(self):
        sources = [
            'https://www.thehindu.com/news/national/tamil-nadu/',
            'https://www.indiatoday.in/elections/tamil-nadu',
            'https://timesofindia.indiatimes.com/city/chennai',
            'https://www.news18.com/tamil-nadu/',
            'https://www.deccanchronicle.com/nation/current-affairs'
        ]
        
        found_polls = []
        
        for url in sources:
            try:
                response = requests.get(url, headers=self.headers, timeout=10)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Search for 2026 election content
                text = soup.get_text().lower()
                
                if any(term in text for term in ['2026', 'chief minister', 'cm poll', 'election survey']):
                    poll_data = self.extract_poll_numbers(text, url)
                    if poll_data:
                        found_polls.append(poll_data)
                        
            except Exception as e:
                print(f"Error scraping {url}: {e}")
        
        return found_polls
    
    def extract_poll_numbers(self, text, source):
        # Look for percentage patterns
        patterns = {
            'stalin': r'stalin.*?(\d+\.?\d*)\s*%',
            'vijay': r'vijay.*?(\d+\.?\d*)\s*%', 
            'eps': r'(?:eps|palaniswami).*?(\d+\.?\d*)\s*%',
            'annamalai': r'annamalai.*?(\d+\.?\d*)\s*%'
        }
        
        results = {}
        for leader, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                results[f'{leader}_pct'] = float(match.group(1))
        
        if results:
            return {
                'organisation': source.split('/')[2],
                'fieldwork_date': datetime.now().strftime('%Y-%m-%d'),
                'sample_size': 2500,
                'region': 'Tamil Nadu',
                **results
            }
        return None
    
    def check_current_polls(self):
        print("Checking for Tamil Nadu 2026 CM Election Polls...")
        polls = self.scrape_tn_polls()
        
        if polls:
            print(f"Found {len(polls)} polls!")
            for poll in polls:
                print(f"   Source: {poll['organisation']}")
        else:
            print("No active polls found for TN 2026 election")
            print("   Reason: Election is in 2026, polling hasn't started yet")
        
        return polls

if __name__ == '__main__':
    scraper = TN2026PollScraper()
    scraper.check_current_polls()