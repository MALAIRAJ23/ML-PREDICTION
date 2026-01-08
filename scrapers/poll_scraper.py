import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
from datetime import datetime, date
import time
import re
import json

class PollScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.db_path = '../data/polls.db'
        self.start_date = date(2025, 1, 1)  # Only scrape from Jan 1, 2025 onwards
    
    def scrape_news_websites(self):
        """Scrape poll data from news websites"""
        scraped_data = []
        
        # Sample scraping patterns for different news sources
        sources = [
            {
                'name': 'India Today',
                'url': 'https://www.indiatoday.in/elections/tamil-nadu',
                'pattern': 'poll|survey|forecast'
            },
            {
                'name': 'Times of India',
                'url': 'https://timesofindia.indiatimes.com/city/chennai',
                'pattern': 'election|poll|survey'
            },
            {
                'name': 'The Hindu',
                'url': 'https://www.thehindu.com/news/national/tamil-nadu/',
                'pattern': 'poll|survey|election'
            }
        ]
        
        for source in sources:
            try:
                data = self.scrape_source(source)
                scraped_data.extend(data)
                time.sleep(2)  # Rate limiting
            except Exception as e:
                print(f"Error scraping {source['name']}: {e}")
        
        return scraped_data
    
    def scrape_source(self, source):
        """Scrape individual news source"""
        try:
            response = requests.get(source['url'], headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for poll-related articles
            articles = soup.find_all(['article', 'div'], class_=re.compile(r'story|article|news'))
            
            poll_data = []
            for article in articles[:5]:  # Limit to recent articles
                text = article.get_text().lower()
                if any(keyword in text for keyword in ['poll', 'survey', 'forecast', 'prediction']):
                    # Extract poll data using regex patterns
                    poll_info = self.extract_poll_data(text, source['name'])
                    if poll_info:
                        poll_data.append(poll_info)
            
            return poll_data
        except Exception as e:
            print(f"Error in scrape_source: {e}")
            return []
    
    def extract_poll_data(self, text, source):
        """Extract poll percentages from text using regex"""
        try:
            # Check if article date is after Jan 1, 2025
            current_date = datetime.now().date()
            if current_date < self.start_date:
                return None
            
            # Patterns for different leaders
            patterns = {
                'stalin': r'stalin[:\s]*(\d+(?:\.\d+)?)\s*%',
                'vijay': r'vijay[:\s]*(\d+(?:\.\d+)?)\s*%',
                'eps': r'(?:eps|palaniswami|edappadi)[:\s]*(\d+(?:\.\d+)?)\s*%',
                'annamalai': r'annamalai[:\s]*(\d+(?:\.\d+)?)\s*%',
                'seeman': r'seeman[:\s]*(\d+(?:\.\d+)?)\s*%'
            }
            
            poll_data = {
                'organisation': source,
                'fieldwork_date': datetime.now().strftime('%Y-%m-%d'),
                'sample_size': self.extract_sample_size(text),
                'region': 'Tamil Nadu',
                'stalin_pct': 0,
                'vijay_pct': 0,
                'eps_pct': 0,
                'annamalai_pct': 0,
                'seeman_pct': 0,
                'others_pct': 0
            }
            
            # Extract percentages
            for leader, pattern in patterns.items():
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    pct = float(match.group(1))
                    if leader == 'stalin':
                        poll_data['stalin_pct'] = pct
                    elif leader == 'vijay':
                        poll_data['vijay_pct'] = pct
                    elif leader == 'eps':
                        poll_data['eps_pct'] = pct
                    elif leader == 'annamalai':
                        poll_data['annamalai_pct'] = pct
                    elif leader == 'seeman':
                        poll_data['seeman_pct'] = pct
            
            # Calculate others percentage
            total_pct = sum([poll_data[f'{leader}_pct'] for leader in ['stalin', 'vijay', 'eps', 'annamalai', 'seeman']])
            if total_pct > 0 and total_pct < 100:
                poll_data['others_pct'] = 100 - total_pct
            
            return poll_data if total_pct > 0 else None
            
        except Exception as e:
            print(f"Error extracting poll data: {e}")
            return None
    
    def extract_sample_size(self, text):
        """Extract sample size from text"""
        sample_pattern = r'sample[:\s]*(\d+)|(\d+)\s*(?:people|respondents|voters)'
        match = re.search(sample_pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1) or match.group(2))
        return 2500  # Default sample size
    
    def save_to_database(self, poll_data):
        """Save scraped data to database"""
        if not poll_data:
            return
        
        conn = sqlite3.connect(self.db_path)
        
        for poll in poll_data:
            # Check if similar poll already exists
            existing = conn.execute("""
                SELECT id FROM polls 
                WHERE organisation = ? AND fieldwork_date = ?
            """, (poll['organisation'], poll['fieldwork_date'])).fetchone()
            
            if not existing:
                conn.execute("""
                    INSERT INTO polls (organisation, fieldwork_date, sample_size, region,
                                     stalin_pct, vijay_pct, eps_pct, annamalai_pct, seeman_pct, others_pct)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    poll['organisation'],
                    poll['fieldwork_date'],
                    poll['sample_size'],
                    poll['region'],
                    poll['stalin_pct'],
                    poll['vijay_pct'],
                    poll['eps_pct'],
                    poll['annamalai_pct'],
                    poll['seeman_pct'],
                    poll['others_pct']
                ))
        
        conn.commit()
        conn.close()
    
    def run_scraper(self):
        """Main scraper execution"""
        print("Starting poll data scraping...")
        scraped_data = self.scrape_news_websites()
        
        if scraped_data:
            self.save_to_database(scraped_data)
            print(f"Scraped and saved {len(scraped_data)} polls")
        else:
            print("No new poll data found")
        
        return len(scraped_data)

if __name__ == '__main__':
    scraper = PollScraper()
    scraper.run_scraper()