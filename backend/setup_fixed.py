#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup script for Tamil Nadu Election Forecasting System
Initializes database and trains ML models
"""

import os
import sys

def setup_project():
    print("=== Tamil Nadu Election Forecasting Setup ===\n")
    
    # Step 1: Initialize enhanced database
    print("1. Initializing enhanced database...")
    try:
        from init_enhanced_db import init_enhanced_database
        init_enhanced_database()
        print("[SUCCESS] Database initialized successfully\n")
    except Exception as e:
        print(f"[ERROR] Database initialization failed: {e}")
        return False
    
    # Step 2: Train ML models
    print("2. Training ML models...")
    try:
        from train_model import ModelTrainer
        trainer = ModelTrainer()
        accuracy = trainer.train_and_save_models()
        print(f"[SUCCESS] Models trained successfully with accuracy: {accuracy:.3f}\n")
    except Exception as e:
        print(f"[ERROR] Model training failed: {e}")
        return False
    
    print("[COMPLETE] Setup completed successfully!")
    print("\nNext steps:")
    print("1. Start backend: python app.py")
    print("2. Start frontend: cd ../frontend && npm start")
    print("3. Access application at http://localhost:3000")
    
    return True

if __name__ == '__main__':
    success = setup_project()
    sys.exit(0 if success else 1)