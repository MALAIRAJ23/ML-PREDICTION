import sqlite3
import os
from datetime import datetime, timedelta
import random

def init_database():
    os.makedirs('../data', exist_ok=True)
    conn = sqlite3.connect('../data/polls.db')
    
    # Create polls table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS polls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            organisation TEXT NOT NULL,
            fieldwork_date DATE NOT NULL,
            sample_size INTEGER,
            region TEXT DEFAULT 'Tamil Nadu',
            stalin_pct REAL DEFAULT 0,
            vijay_pct REAL DEFAULT 0,
            eps_pct REAL DEFAULT 0,
            annamalai_pct REAL DEFAULT 0,
            seeman_pct REAL DEFAULT 0,
            others_pct REAL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Insert sample data for testing
    sample_polls = [
        ('C-Voter', '2024-01-15', 2500, 'Tamil Nadu', 42.5, 18.2, 22.1, 8.3, 4.2, 4.7),
        ('ABP-CVoter', '2024-01-20', 3200, 'Tamil Nadu', 41.8, 19.5, 21.3, 9.1, 3.8, 4.5),
        ('India Today-Axis', '2024-02-01', 2800, 'Tamil Nadu', 43.2, 17.8, 23.4, 7.9, 4.1, 3.6),
        ('Times Now-VMR', '2024-02-10', 2100, 'Tamil Nadu', 40.9, 20.1, 22.8, 8.7, 3.9, 3.6),
        ('Republic-P-MARQ', '2024-02-15', 2650, 'Tamil Nadu', 44.1, 18.9, 21.2, 8.2, 4.3, 3.3),
        ('News18-IPSOS', '2024-02-25', 3100, 'Tamil Nadu', 42.7, 19.8, 22.5, 7.8, 3.7, 3.5),
        ('ABP News-CVoter', '2024-03-05', 2900, 'Tamil Nadu', 43.8, 18.4, 21.9, 8.5, 4.0, 3.4),
        ('India TV-CNX', '2024-03-12', 2400, 'Tamil Nadu', 41.5, 20.3, 23.1, 8.9, 3.8, 2.4),
        ('Zee News-DESIGNBOXED', '2024-03-18', 2750, 'Tamil Nadu', 44.3, 17.9, 21.7, 8.1, 4.2, 3.8),
        ('CNN-News18', '2024-03-25', 3300, 'Tamil Nadu', 42.9, 19.2, 22.3, 8.6, 3.9, 3.1)
    ]
    
    conn.executemany('''
        INSERT INTO polls (organisation, fieldwork_date, sample_size, region,
                         stalin_pct, vijay_pct, eps_pct, annamalai_pct, seeman_pct, others_pct)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', sample_polls)
    
    conn.commit()
    conn.close()
    print("Database initialized successfully!")

if __name__ == '__main__':
    init_database()