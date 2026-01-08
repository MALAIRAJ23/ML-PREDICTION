import sqlite3
import os
from datetime import datetime, timedelta
import random

def init_enhanced_database():
    os.makedirs('../data', exist_ok=True)
    conn = sqlite3.connect('../data/polls.db')
    
    # Drop existing table and recreate
    conn.execute('DROP TABLE IF EXISTS polls')
    
    # Create polls table
    conn.execute('''
        CREATE TABLE polls (
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
    
    # Enhanced sample data with more diversity
    organisations = [
        'C-Voter', 'ABP-CVoter', 'India Today-Axis', 'Times Now-VMR', 
        'Republic-P-MARQ', 'News18-IPSOS', 'ABP News-CVoter', 'India TV-CNX',
        'Zee News-DESIGNBOXED', 'CNN-News18', 'CSDS-Lokniti', 'Jan Ki Baat'
    ]
    
    regions = ['Tamil Nadu', 'Chennai', 'Coimbatore', 'Madurai', 'Salem', 'Tiruchirappalli']
    
    # Generate diverse poll data over time
    sample_polls = []
    base_date = datetime(2024, 1, 1)
    
    for i in range(50):  # Generate 50 diverse polls
        date = base_date + timedelta(days=i*7)  # Weekly polls
        org = random.choice(organisations)
        region = random.choice(regions)
        sample_size = random.randint(1500, 4000)
        
        # Create realistic but diverse poll results
        # Different scenarios based on time progression
        if i < 15:  # Early polls - Stalin leading
            stalin = random.uniform(38, 48)
            vijay = random.uniform(15, 25)
            eps = random.uniform(18, 28)
            annamalai = random.uniform(6, 12)
            seeman = random.uniform(2, 8)
        elif i < 30:  # Mid period - More competitive
            stalin = random.uniform(35, 45)
            vijay = random.uniform(18, 28)
            eps = random.uniform(15, 25)
            annamalai = random.uniform(8, 15)
            seeman = random.uniform(3, 10)
        else:  # Later polls - Vijay gaining
            stalin = random.uniform(32, 42)
            vijay = random.uniform(22, 32)
            eps = random.uniform(12, 22)
            annamalai = random.uniform(6, 14)
            seeman = random.uniform(2, 8)
        
        # Ensure percentages sum to ~100
        total = stalin + vijay + eps + annamalai + seeman
        others = max(0, 100 - total)
        
        # Normalize to exactly 100%
        factor = 100 / (total + others)
        stalin *= factor
        vijay *= factor
        eps *= factor
        annamalai *= factor
        seeman *= factor
        others *= factor
        
        sample_polls.append((
            org, date.strftime('%Y-%m-%d'), sample_size, region,
            round(stalin, 1), round(vijay, 1), round(eps, 1), 
            round(annamalai, 1), round(seeman, 1), round(others, 1)
        ))
    
    # Add some polls where other candidates lead (for diversity)
    special_polls = [
        ('Special Survey-1', '2024-06-15', 2800, 'Chennai', 28.5, 35.2, 20.1, 8.3, 4.2, 3.7),  # Vijay leading
        ('Special Survey-2', '2024-07-01', 3200, 'Coimbatore', 32.1, 25.8, 30.4, 6.9, 3.1, 1.7),  # EPS leading
        ('Special Survey-3', '2024-07-15', 2100, 'Madurai', 29.8, 28.9, 25.2, 12.1, 2.8, 1.2),  # Close race
        ('Special Survey-4', '2024-08-01', 2650, 'Salem', 31.5, 31.8, 22.3, 9.4, 3.5, 1.5),  # Vijay slight lead
        ('Special Survey-5', '2024-08-15', 3100, 'Tiruchirappalli', 35.2, 29.1, 19.8, 10.2, 4.1, 1.6),  # Stalin comeback
    ]
    
    sample_polls.extend(special_polls)
    
    conn.executemany('''
        INSERT INTO polls (organisation, fieldwork_date, sample_size, region,
                         stalin_pct, vijay_pct, eps_pct, annamalai_pct, seeman_pct, others_pct)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', sample_polls)
    
    conn.commit()
    conn.close()
    print(f"Enhanced database initialized with {len(sample_polls)} diverse polls!")

if __name__ == '__main__':
    init_enhanced_database()