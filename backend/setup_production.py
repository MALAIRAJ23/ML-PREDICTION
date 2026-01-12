#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production Setup Script with Real Data
Uses realistic poll data instead of synthetic data
"""

import os
import sys

def setup_production():
    print("=== Tamil Nadu Election Forecasting Setup (Production) ===\n")
    
    # Step 1: Run real data pipeline
    print("1. Setting up with realistic poll data...")
    try:
        from real_data_pipeline_fixed import RealDataMLPipeline
        pipeline = RealDataMLPipeline()
        success = pipeline.run_full_pipeline()
        
        if success:
            print("[SUCCESS] Production setup completed successfully\n")
        else:
            print("[ERROR] Production setup failed")
            return False
            
    except Exception as e:
        print(f"[ERROR] Setup failed: {e}")
        return False
    
    print("[COMPLETE] Production setup completed!")
    print("\nFeatures:")
    print("- Real ML predictions using Random Forest")
    print("- Realistic poll data (8+ recent polls)")
    print("- Web scraping capability (when available)")
    print("- Automatic model retraining")
    print("\nAPI Endpoints:")
    print("- GET /api/forecast - ML predictions")
    print("- POST /api/train - Retrain model")
    print("- GET /api/polls - Poll data")
    print("\nNext steps:")
    print("1. Start backend: python app.py")
    print("2. Start frontend: cd ../frontend && npm start")
    print("3. Access application at http://localhost:3000")
    
    return True

if __name__ == '__main__':
    success = setup_production()
    sys.exit(0 if success else 1)