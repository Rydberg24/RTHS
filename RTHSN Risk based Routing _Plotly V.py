import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import random, time, re
from faker import Faker
from concurrent.futures import ThreadPoolExecutor, as_completed
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# 1. DATASET GENERATION (8 CATEGORIES, REALISTIC US DUTIES)
fake = Faker()

CATEGORY_INFO = {
    'Electronics/Laptops': {
        'hts': '8471.30.0100', 'duty': 0.0, 'avg_value': 800.0,
        'reg_mult': 1.0, 'scrutiny': 0.5, 'perishable': False
    },
    'Apparel/Menswear': {
        'hts': '6109.10.0014', 'duty': 0.165, 'avg_value': 30.0,
        'reg_mult': 1.0, 'scrutiny': 0.7, 'perishable': False
    },
    'Beverages/Bottled Water': {
        'hts': '2201.10.0000', 'duty': 0.0, 'avg_value': 0.50,
        'reg_mult': 1.0, 'scrutiny': 0.3, 'perishable': True
    },
    'Furniture/Office Chairs': {
        'hts': '9401.30.0000', 'duty': 0.08, 'avg_value': 400.0,
        'reg_mult': 1.0, 'scrutiny': 0.5, 'perishable': False
    },
    'Pharmaceuticals/OTC': {
        'hts': '3004.90.0000', 'duty': 0.0, 'avg_value': 15.0,
        'reg_mult': 5.0, 'scrutiny': 1.0, 'perishable': True
    },
    'Automotive/Brake Pads': {
        'hts': '8708.30.0000', 'duty': 0.025, 'avg_value': 40.0,
        'reg_mult': 1.0, 'scrutiny': 0.8, 'perishable': False
    },
    'Toys/Board Games': {
        'hts': '9504.90.6000', 'duty': 0.0, 'avg_value': 25.0,
        'reg_mult': 1.0, 'scrutiny': 0.5, 'perishable': False
    },
    'Canned Foods': {
        'hts': '2005.51.0000', 'duty': 0.175, 'avg_value': 2.50,
        'reg_mult': 1.0, 'scrutiny': 0.6, 'perishable': True
    }
}

def generate_wms_dataset(n_rows=10000):
    categories_products = {
        'Electronics/Laptops': ['Dell Latitude i5', 'HP Pavilion 15', 'Lenovo ThinkPad'],
        'Apparel/Menswear': ['Gildan T-Shirt M', 'Levi Jeans 32x34', 'Polo Shirt L'],
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
        weight = round(random.uniform(0.01, 2.5), 3) if random.random() > 0.15 else None
        
        # Lot expiry
        if info['perishable']:
            lot = fake.date_between(start_date='-3M', end_date='+12M')
            if random.random() < 0.1:
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
            'Correct_HTS': info['hts']
        }
        data.append(row)
    return pd.DataFrame(data)

df = generate_wms_dataset(10000)


# 2. MOCK ATLAS CLASSIFIER

def mock_atlas_classify(product_desc, category, weight_kg):
    if re.match(r'^[A-Z0-9\s\-\.]+$', product_desc) and any(c.isdigit() for c in product_desc):
        base_conf = 0.55   # part-number style
    elif any(word in product_desc.lower() for word in ['game', 'chair', 'toy', 'water', 'laptop']):
        base_conf = 0.85   # clear consumer keywords
    else:
        base_conf = 0.75
        
    confidence = np.clip(np.random.normal(base_conf, 0.12), 0.3, 0.99)
    correct = CATEGORY_INFO[category]['hts']
    
    if confidence < 0.65 and random.random() < 0.3:
        wrong_codes = [v['hts'] for v in CATEGORY_INFO.values() if v['hts'] != correct]
        hts_code = random.choice(wrong_codes) if wrong_codes else correct
    else:
        hts_code = correct
        
    time.sleep(0.02)
    return hts_code, round(confidence, 3)


# 3. EXTENDED RISK SCORING & RECALIBRATED THRESHOLDS

def compute_risk_total(confidence, declared_value, duty_rate, p_reg, r_reg,
                       pallet_qty, lot_expiry_str, perishable, max_pallet=1200):
    uncertainty = 1 - confidence
    exposure = declared_value * duty_rate + p_reg * r_reg
    vol_factor = min(pallet_qty / max_pallet, 1.0)
    exp_factor = 1.0
    
    if perishable and lot_expiry_str != 'N/A':
        try:
            exp_date = datetime.strptime(lot_expiry_str, '%Y-%m-%d').date()
            days_left = (exp_date - date.today()).days
            if days_left < 90:
                exp_factor = 1 + max(0, 1 - days_left/90.0)
        except:
            pass
            
    return uncertainty * exposure * vol_factor * exp_factor

def assign_tier(risk_score):
    if risk_score <= 0.20:
        return 'GREEN'
    elif risk_score <= 1.00:
        return 'YELLOW'
    else:
        return 'RED'

# 4. BATCH PIPELINE (10-worker thread pool)

def run_risk_pipeline(df):
    print("Processing 10,000 records with extended risk formula...")
    df = df.copy()
    df['HTS_Code'] = ''
    df['Confidence'] = 0.0
    df['Risk_Total'] = 0.0
    df['Risk_Tier'] = ''
    total_start = time.time()
    
    for i in range(0, len(df), 100):
        batch = df.iloc[i:i+100]
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
        
    print(f"\nCompleted in {time.time()-total_start:.1f}s\n")
    return df

df_risk = run_risk_pipeline(df)

# 5. INTERACTIVE DASHBOARD (Plotly figures)

# 5.1 Risk Score Distribution (histogram with staggered threshold lines)
fig1 = px.histogram(
    df_risk, x='Risk_Total', nbins=40,
    title='Total Risk Score Distribution',
    labels={'Risk_Total': 'R_total'},
    color_discrete_sequence=['steelblue'],
    opacity=0.8
)
# Fix overlapping text by sending 0.20 to the left and 1.00 to the right
fig1.add_vline(x=0.20, line_dash='dash', line_color='orange',
               annotation_text='GREEN/YELLOW (0.20)', annotation_position='top left')
fig1.add_vline(x=1.00, line_dash='dash', line_color='red',
               annotation_text='YELLOW/RED (1.00)', annotation_position='top right')
fig1.update_layout(bargap=0.1)
fig1.show()

# 5.2 Risk Tier Pie (fixed colour order)
tier_order = ['GREEN', 'YELLOW', 'RED']
tier_counts = df_risk['Risk_Tier'].value_counts().reindex(tier_order, fill_value=0)
color_map = {'GREEN': '#2ca02c', 'YELLOW': '#ff7f0e', 'RED': '#d62728'}

fig2 = go.Figure(data=[go.Pie(
    labels=tier_order, values=tier_counts.values,
    marker=dict(colors=[color_map[t] for t in tier_order]),
    hole=0.3, sort=False
)])
fig2.update_layout(title='Risk-Based Put-Away Allocation')
fig2.show()

# 5.3 Risk Score by Category (boxplot)
order = df_risk.groupby('Category_Subcategory')['Risk_Total'].median().sort_values(ascending=False).index
fig3 = px.box(
    df_risk, x='Category_Subcategory', y='Risk_Total',
    category_orders={'Category_Subcategory': order.tolist()},
    title='Risk Score by Product Category',
    labels={'Risk_Total': 'R_total'},
    color_discrete_sequence=['steelblue']
)
fig3.add_hline(y=0.20, line_dash='dash', line_color='orange', opacity=0.7)
fig3.add_hline(y=1.00, line_dash='dash', line_color='red', opacity=0.7)
fig3.update_xaxes(tickangle=45)
fig3.show()

# 5.4 Confidence by Category (boxplot)
conf_order = df_risk.groupby('Category_Subcategory')['Confidence'].median().sort_values().index
fig4 = px.box(
    df_risk, x='Category_Subcategory', y='Confidence',
    category_orders={'Category_Subcategory': conf_order.tolist()},
    title='Simulated ATLAS Confidence by Category',
    labels={'Confidence': 'Confidence'},
    color_discrete_sequence=['steelblue']
)
fig4.add_hline(y=0.75, line_dash='dash', line_color='orange',
               annotation_text='Median baseline (0.75)')
fig4.update_xaxes(tickangle=45)
fig4.show()


# 5.5 Combined dashboard (2x2)

from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig_comb = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Risk Score Distribution', 'Risk Tier Allocation',
                    'Risk Score by Category', 'Confidence by Category'),
    specs=[[{'type': 'xy'},    {'type': 'domain'}],    # row 1: xy + pie
           [{'type': 'xy'},    {'type': 'xy'}]]        # row 2: both xy
)

# Histogram (row 1, col 1)
hist_trace = go.Histogram(x=df_risk['Risk_Total'], nbinsx=40,
                          marker_color='steelblue', showlegend=False)
fig_comb.add_trace(hist_trace, row=1, col=1)

# Pie (row 1, col 2)
pie_trace = go.Pie(labels=tier_order, values=tier_counts.values,
                   marker=dict(colors=[color_map[t] for t in tier_order]),
                   sort=False, showlegend=False)
fig_comb.add_trace(pie_trace, row=1, col=2)

# Box risk by category (row 2, col 1)
for cat in order:
    cat_data = df_risk[df_risk['Category_Subcategory'] == cat]['Risk_Total']
    fig_comb.add_trace(go.Box(y=cat_data, name=cat, marker_color='steelblue',
                              showlegend=False), row=2, col=1)

# Box confidence by category (row 2, col 2)
for cat in conf_order:
    cat_data = df_risk[df_risk['Category_Subcategory'] == cat]['Confidence']
    fig_comb.add_trace(go.Box(y=cat_data, name=cat, marker_color='steelblue',
                              showlegend=False), row=2, col=2)


# Subplot 1 (Histogram) vertical lines 
fig_comb.add_shape(type="line", x0=0.20, x1=0.20, y0=0, y1=1, xref="x", yref="y domain", 
                   line=dict(color="orange", dash="dash"))
fig_comb.add_annotation(x=0.20, y=1.05, text="(0.20)", showarrow=False, xanchor="right", xref="x", yref="y domain")

fig_comb.add_shape(type="line", x0=1.00, x1=1.00, y0=0, y1=1, xref="x", yref="y domain", 
                   line=dict(color="red", dash="dash"))
fig_comb.add_annotation(x=1.00, y=1.05, text="(1.00)", showarrow=False, xanchor="left", xref="x", yref="y domain")

# Subplot 3 (Risk Boxplot) horizontal lines 
fig_comb.add_shape(type="line", x0=0, x1=1, y0=0.20, y1=0.20, xref="x2 domain", yref="y2", 
                   line=dict(color="orange", dash="dash"))
fig_comb.add_shape(type="line", x0=0, x1=1, y0=1.00, y1=1.00, xref="x2 domain", yref="y2", 
                   line=dict(color="red", dash="dash"))

# Subplot 4 (Confidence Boxplot) horizontal line 
fig_comb.add_shape(type="line", x0=0, x1=1, y0=0.75, y1=0.75, xref="x3 domain", yref="y3", 
                   line=dict(color="orange", dash="dash"))

# Final Layout
fig_comb.update_layout(height=800, width=1000,
                       title_text="RTHSN Framework Interactive Dashboard")
fig_comb.update_xaxes(tickangle=45, row=2, col=1)
fig_comb.update_xaxes(tickangle=45, row=2, col=2)
fig_comb.show()

print("\nClick on any chart to explore.")