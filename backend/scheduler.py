import schedule
import time
import sys
import os

# Add the scrapers directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scrapers'))

from poll_scraper import PollScraper

def run_scraper():
    """Run the poll scraper"""
    print(f"Starting scheduled scraping at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    scraper = PollScraper()
    try:
        count = scraper.run_scraper()
        print(f"Scraping completed. Found {count} new polls.")
    except Exception as e:
        print(f"Scraping failed: {e}")

def main():
    print("Tamil Nadu Election Poll Scheduler Started")
    print("Scraping every 6 hours...")
    
    # Schedule scraping every 6 hours
    schedule.every(6).hours.do(run_scraper)
    
    # Run once immediately
    run_scraper()
    
    # Keep the scheduler running
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

if __name__ == '__main__':
    main()