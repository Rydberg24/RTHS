
# RTHSN FRAMEWORK – Proof‑of‑Concept

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta, date
import random
import time
import re
from faker import Faker
from concurrent.futures import ThreadPoolExecutor, as_completed

# Plot settings
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

# 1. DATASET GENERATION (8 CATEGORIES)
fake = Faker()

# Category metadata for simulation
CATEGORY_INFO = {
    'Electronics/Laptops': {
        'hts': '8471.30.0100',
        'duty': 0,
        'avg_value': 800.0,   # EUR per unit
        'reg_mult': 1.0,      # P_reg
        'scrutiny': 0.5,      # r_reg
        'perishable': False   # Expiry 
    },
    'Apparel/Menswear': {
        'hts': '6109.10.0014',
        'duty': 0.165,   
        'avg_value': 30.0,
        'reg_mult': 1.0,
        'scrutiny': 0.7,
        'perishable': False
    },
    'Beverages/Bottled Water': {
        'hts': '2201.10.0000',
        'duty': 0,
        'avg_value': 0.50,
        'reg_mult': 1.0,
        'scrutiny': 0.3,
        'perishable': True
    },
    'Furniture/Office Chairs': {
        'hts': '9401.30.0000',
        'duty': 0.075,
        'avg_value': 400.0,
        'reg_mult': 1.0,
        'scrutiny': 0.5,
        'perishable': False
    },
    'Pharmaceuticals/OTC': {
        'hts': '3004.90.0000',
        'duty': 0,
        'avg_value': 15.0,
        'reg_mult': 5.0,      
        'scrutiny': 1.0,
        'perishable': True
    },
    'Automotive/Brake Pads': {
        'hts': '8708.30.0000',
        'duty': 0.025,
        'avg_value': 40.0,
        'reg_mult': 1.0,
        'scrutiny': 0.8,
        'perishable': False
    },
    'Toys/Board Games': {
        'hts': '9504.90.6000',
        'duty': 0,
        'avg_value': 25.0,
        'reg_mult': 1.0,
        'scrutiny': 0.5,
        'perishable': False
    },
    'Canned Foods': {
        'hts': '2005.51.0000',
        'duty': 0.175,
        'avg_value': 2.50,
        'reg_mult': 1.0,
        'scrutiny': 0.7,
        'perishable': True
    }
}

def generate_wms_dataset(n_rows=10000):
    """Generating realistic Walmart-style inbound scan synthetic dataset with 8 categories."""
    categories_products = {
        'Electronics/Laptops': ['Dell Latitude i5', 'HP Pavilion 15', 'Lenovo ThinkPad'],
        'Apparel/Menswear': ['Zara T-Shirt M', 'Levi Jeans 32x34', 'Polo Shirt L'],
        'Beverages/Bottled Water': ['Nestle Pure Life 500ml', 'Dasani 16oz', 'Aquafina 1L'],
        'Furniture/Office Chairs': ['Herman Miller Aeron', 'IKEA Markus', 'Steelcase Leap'],
        'Pharmaceuticals/OTC': ['Tylenol 500mg', 'Advil 200mg', 'Benadryl Allergy'],
        'Automotive/Brake Pads': ['BOSCH 0 986 494 411', 'ATE 13.0460-7258.2', 'TRW GDB 350'],
        'Toys/Board Games': ['Monopoly Classic', 'Catan', 'Uno Card Game'],
        'Canned Foods': ['Heinz Baked Beans', 'Campbells Tomato Soup', 'Del Monte Corn']
    }
    
    timestamps = pd.date_range('2026-02-17 14:30:00', periods=n_rows, freq='30S')
    data = []
    for i, ts in enumerate(timestamps):
        cat = random.choice(list(categories_products.keys()))
        prod = random.choice(categories_products[cat])
        info = CATEGORY_INFO[cat]
        
        # Realistic imperfections
        weight = round(random.uniform(0.01, 2.5), 3) if random.random() > 0.15 else None
        # Lot expiry
        if info['perishable']:
            lot = fake.date_between(start_date='-3M', end_date='+12M')
            if random.random() < 0.1:   # expired
                lot = fake.date_between(start_date='-12M', end_date='-1M')
        else:
            lot = fake.date_between(start_date='-12M', end_date='+24M')
        lot_str = 'N/A' if random.random() > 0.7 else lot.strftime('%Y-%m-%d')
        
        row = {
            'Timestamp_UTC': ts,
            'Barcode_ID': f'WM-BC{str(i+1).zfill(6)}',
            'Vendor_SKU': f"{prod.replace(' ', '').upper()[:8]}{random.randint(100,999)}",
            'Walmart_Item_ID': str(random.randint(1000000000, 9999999999)),
            'Product_Description': prod,
            'GTIN_UPC': f"{random.randint(1000000,9999999):07d}{random.randint(100000,999999):06d}",
            'Category_Subcategory': cat,
            'Weight_kg': weight,
            'Pallet_Qty': random.randint(24, 1200) if 'Pharma' not in cat else random.randint(12, 48),
            'Bin_Location': f"{random.choice('ABCDEF')}-{random.randint(1,10):02d}-{random.randint(1,10):02d}",
            'Status': 'RECEIVED',
            'Lot_Expiry': lot_str,
            'Declared_Value': info['avg_value'],
            'P_reg': info['reg_mult'],
            'r_reg': info['scrutiny'],
            'Duty_Rate': info['duty'],
            'Correct_HTS': info['hts']   # ground truth for reference
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv('wms_inbound_10k_scans_extended.csv', index=False)
    print(f"Generated {len(df)} records → wms_inbound_10k_scans_extended.csv")
    return df

# Generate dataset
df = generate_wms_dataset(10000)

# 2.  MOCK ATLAS CLASSIFIER
def mock_atlas_classify(product_desc, category, weight_kg):
    """
    Simulates ATLAS classification with realistic confidence.
    - High confidence for clear consumer goods
    - Low confidence for technical part number like descriptions
    """
    # Determine base confidence from description complexity
    if re.match(r'^[A-Z0-9\s\-\.]+$', product_desc) and any(c.isdigit() for c in product_desc):
        base_conf = 0.55   # part numbers (automotive)
    elif any(word in product_desc.lower() for word in ['game', 'chair', 'toy', 'water', 'laptop']):
        base_conf = 0.85   # clear consumer descriptions
    else:
        base_conf = 0.75
    
    confidence = np.clip(np.random.normal(base_conf, 0.12), 0.3, 0.99)
    
    # Simulate HTS output: mostly correct, but 30% chance of plausible error when confidence < 0.65
    correct = CATEGORY_INFO[category]['hts']
    if confidence < 0.65 and random.random() < 0.3:
        # pick a random wrong HTS code from other categories
        wrong_codes = [v['hts'] for v in CATEGORY_INFO.values() if v['hts'] != correct]
        hts_code = random.choice(wrong_codes) if wrong_codes else correct
    else:
        hts_code = correct
    
    time.sleep(0.02)   # simulated API latency
    return hts_code, round(confidence, 3)

# 3. RISK SCORING (Multi‑Factor)
def compute_risk_total(confidence, declared_value, duty_rate, p_reg, r_reg,
                       pallet_qty, lot_expiry_str, perishable, max_pallet=1200):
    """R_total = (1 - c) * (V*d + P_reg * r_reg) * vol_factor * exp_factor"""
    uncertainty = 1 - confidence
    # Financial + regulatory exposure per unit
    exposure = (declared_value * duty_rate) + (p_reg * r_reg)
    # Volume factor (normalised)
    vol_factor = min(pallet_qty / max_pallet, 1.0)
    # Expiry factor (for perishables with near‑term expiry)
    exp_factor = 1.0
    if perishable and lot_expiry_str != 'N/A':
        try:
            exp_date = datetime.strptime(lot_expiry_str, '%Y-%m-%d').date()
            today = date.today()
            days_left = (exp_date - today).days
            if days_left < 90:
                exp_factor = 1 + max(0, 1 - days_left/90.0)
        except:
            pass
    
    risk = uncertainty * exposure * vol_factor * exp_factor
    return risk

def assign_tier(risk_score):
    if risk_score <= 0.20:
        return 'GREEN'
    elif risk_score <= 1.00:
        return 'YELLOW'
    else:
        return 'RED'

# 4. BATCH PROCESSING PIPELINE 
def run_risk_pipeline(df):
    print("RTHSN Extended Pipeline processing 10K records...")
    df = df.copy()
    df['HTS_Code'] = ''
    df['Confidence'] = 0.0
    df['Risk_Total'] = 0.0
    df['Risk_Tier'] = ''
    
    total_start = time.time()
    batch_size = 100
    
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {}
            for idx, row in batch.iterrows():
                desc = row['Product_Description']
                cat = row['Category_Subcategory']
                wgt = row['Weight_kg'] if pd.notna(row['Weight_kg']) else 0.5
                futures[executor.submit(mock_atlas_classify, desc, cat, wgt)] = idx
            
            for future in as_completed(futures):
                idx = futures[future]
                hts_code, confidence = future.result()
                df.at[idx, 'HTS_Code'] = hts_code
                df.at[idx, 'Confidence'] = confidence
                
                row = df.loc[idx]
                risk = compute_risk_total(
                    confidence=confidence,
                    declared_value=row['Declared_Value'],
                    duty_rate=row['Duty_Rate'],
                    p_reg=row['P_reg'],
                    r_reg=row['r_reg'],
                    pallet_qty=row['Pallet_Qty'],
                    lot_expiry_str=row['Lot_Expiry'],
                    perishable=CATEGORY_INFO[row['Category_Subcategory']]['perishable']
                )
                df.at[idx, 'Risk_Total'] = round(risk, 4)
                df.at[idx, 'Risk_Tier'] = assign_tier(risk)
        
        elapsed = time.time() - total_start
        print(f"  {min(i+100, len(df))}/{len(df)} – {elapsed:.1f}s", end='\r')
    
    total_time = time.time() - total_start
    print(f"\n Complete! {len(df)} records in {total_time:.2f}s\n")
    return df

df_risk = run_risk_pipeline(df)

# 5. RESULTS SUMMARY
print("RISK BASED TIER DISTRIBUTION")
tier_counts = df_risk['Risk_Tier'].value_counts()
for tier in ['GREEN', 'YELLOW', 'RED']:
    cnt = tier_counts.get(tier, 0)
    print(f"  {tier}: {cnt} ({cnt/len(df_risk)*100:.1f}%)")

print("\n Average Risk Score by Category:")
print(df_risk.groupby('Category_Subcategory')['Risk_Total'].mean().sort_values(ascending=False).round(4))

# 6. VISUALISATION
# 6.1 Risk Score Distribution
plt.figure(figsize=(8,5))
sns.histplot(df_risk['Risk_Total'], bins=30, color='steelblue', edgecolor='white')
plt.axvline(0.02, color='orange', linestyle='--', label='GREEN/YELLOW (R=0.02)')
plt.axvline(0.10, color='red', linestyle='--', label='YELLOW/RED (R=0.10)')
plt.title('Total Risk Score Distribution', fontsize=14)
plt.xlabel('R_total')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.savefig('fig_extended_risk_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# 6.2 Tier Pie Chart
tier_order = ['GREEN', 'YELLOW', 'RED']
tier_counts_fixed = df_risk['Risk_Tier'].value_counts().reindex(tier_order, fill_value=0)
colors = {'GREEN': '#2ca02c', 'YELLOW': '#ff7f0e', 'RED': '#d62728'}

plt.figure(figsize=(6,6))
plt.pie(tier_counts_fixed,
        labels=tier_order,
        autopct='%1.1f%%',
        startangle=90,
        colors=[colors[t] for t in tier_order],
        explode=(0.05, 0, 0),
        shadow=True)
plt.title('Risk Based Put Away Allocation', fontsize=14)
plt.tight_layout()
plt.savefig('fig_extended_risk_tier_pie.png', dpi=300, bbox_inches='tight')
plt.show()

# 6.3 Risk Score by Category (Boxplot)
order = df_risk.groupby('Category_Subcategory')['Risk_Total'].median().sort_values(ascending=False).index
plt.figure(figsize=(12,6))
sns.boxplot(data=df_risk, x='Category_Subcategory', y='Risk_Total', order=order)
plt.xticks(rotation=45, ha='right')
plt.axhline(0.02, color='orange', linestyle='--', alpha=0.5)
plt.axhline(0.10, color='red', linestyle='--', alpha=0.5)
plt.title('Risk Score by Product Category', fontsize=14)
plt.ylabel('R_total')
plt.tight_layout()
plt.savefig('fig_extended_risk_by_category.png', dpi=300, bbox_inches='tight')
plt.show()

# 6.4 Confidence by Category (Boxplot)
plt.figure(figsize=(12,6))
conf_order = df_risk.groupby('Category_Subcategory')['Confidence'].median().sort_values().index
sns.boxplot(data=df_risk, x='Category_Subcategory', y='Confidence', order=conf_order)
plt.xticks(rotation=45, ha='right')
plt.title('Simulated ATLAS Confidence by Category', fontsize=14)
plt.ylabel('Confidence')
plt.tight_layout()
plt.savefig('fig_extended_confidence_by_category.png', dpi=300, bbox_inches='tight')
plt.show()

df_risk.to_csv('wms_10k_extended_risk_classified.csv', index=False)
print("Classified dataset saved → wms_10k_extended_risk_classified.csv")