#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Setup Script with Real Data Integration
Initializes database with real scraped data and trains ML models
"""

import os
import sys

def setup_with_real_data():
    print("=== Tamil Nadu Election Forecasting Setup (Real Data) ===\n")
    
    # Step 1: Run real data pipeline
    print("1. Running real data pipeline...")
    try:
        from real_data_pipeline import RealDataMLPipeline
        pipeline = RealDataMLPipeline()
        success = pipeline.run_full_pipeline()
        
        if success:
            print("[SUCCESS] Real data pipeline completed successfully\n")
        else:
            print("[ERROR] Real data pipeline failed")
            return False
            
    except Exception as e:
        print(f"[ERROR] Pipeline execution failed: {e}")
        return False
    
    print("[COMPLETE] Setup completed with real data!")
    print("\nData Sources:")
    print("- Web scraped poll data (if available)")
    print("- Realistic fallback data (if scraping fails)")
    print("- ML model trained on actual poll trends")
    print("\nNext steps:")
    print("1. Start backend: python app.py")
    print("2. Start frontend: cd ../frontend && npm start")
    print("3. Access application at http://localhost:3000")
    print("4. Use /api/scrape_and_retrain to update with fresh data")
    
    return True

if __name__ == '__main__':
    success = setup_with_real_data()
    sys.exit(0 if success else 1)