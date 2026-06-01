
# RTHSN Framework – Real Time HSN Classification framework for warehouse handling systems using LLM Integration

This repository contains the complete Python implementation of the **Real‑Time HSN Classification Framework (RTHSN)** introduced in the research paper:

> *“A Real‑Time HSN Classification Framework for Warehouse Inbound Operations: Integrating Large Language Models with Material Handling Decision Systems”*  

The framework demonstrates how Large Language Model (LLM) powered Harmonized System Nomenclature (HSN/HTS) classification can be integrated directly into a warehouse inbound receiving workflow – at the barcode scan point – enabling risk‑based put‑away decisions before any storage commitment.

## Repository Contents

| File | Description |
|------|-------------|
| `RTHSN Risk based Routing.py` | Main simulation script: generates synthetic dataset (10,000 scans, 8 product categories), runs the classification pipeline (mock ATLAS LLM), computes multi‑factor risk scores, assigns GREEN/YELLOW/RED tiers, and produces static visualisations (matplotlib) and CSV outputs. |
| `RTHSN Risk based Routing_Plotly V.py` | Interactive dashboard version: after running the simulation (or loading an existing classified CSV), this script launches an interactive Plotly dashboard with filters, histograms, boxplots, and a data table for exploring results. |
| `README.md` | This file. |

## Key Features

- **8 product categories** with realistic US HTS duty rates, unit values, regulatory penalty multipliers, and perishability flags.
- **Stochastic mock ATLAS classifier** – confidence varies based on description clarity (consumer terms → high confidence, part numbers → low confidence); includes realistic error simulation.
- **Multi‑factor risk score**  
```math
\displaystyle
R_{\text{total}}
=
(1-c)
\times
\left(
Vd + P_{\text{reg}} r_{\text{reg}}
\right)
\times
\text{vol}_{\text{factor}}
\times
\text{exp}_{\text{factor}}
```

$$
\text{vol.factor} = \min\left(\frac{\text{pallet.qty}}{1200}, 1.0\right)
$$

$$
\text{exp.factor} = 
\begin{cases}
1 + \max\left(0, 1 - \frac{\text{days.left}}{90}\right) & \text{for perishables with } <90 \text{ days to expiry} \\
1.0 & \text{otherwise}
\end{cases}
$$
- **Three‑tier risk‑based routing**  
  - **GREEN** (R ≤ 0.20) → direct to production bin  
  - **YELLOW** (0.20 < R ≤ 1.00) → staging lane + spot‑check  
  - **RED** (R > 1.00) → quarantine + full manual review
- **Parallel batch processing** (10 workers) – processes 10,000 records in ~55 seconds.
- **Interactive Plotly dashboard** – filter by tier, category, confidence, expiry; explore distributions dynamically.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/RTHSN.git
   cd RTHSN
   ```

2. (Optional) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\Scripts\activate      # Windows
   ```

3. Install required packages:
   ```bash
   pip install pandas numpy matplotlib seaborn plotly dash faker
   ```

   *Note: `dash` is only needed for the interactive dashboard script.*

## Usage

### 1. Run the main simulation (static outputs)

```bash
python "RTHSN Risk based Routing.py"
```

This script will:
- Generate `wms_inbound_10k_scans_extended.csv` – raw synthetic inbound scan data.
- Run the risk pipeline (mock classification + risk scoring).
- Save `wms_10k_extended_risk_classified.csv` – final results with HTS codes, confidence scores, risk totals, and tiers.
- Produce four static figures (saved as PNG in the current directory):
  - `fig_extended_risk_distribution.png` – histogram of total risk score.
  - `fig_extended_risk_tier_pie.png` – pie chart of tier allocation.
  - `fig_extended_risk_by_category.png` – boxplots of risk by product category.
  - `fig_extended_confidence_by_category.png` – boxplots of simulated confidence by category.
- Print a summary of tier distribution and average risk per category to the console.

**Typical output (example):**
```
RISK BASED TIER DISTRIBUTION
  GREEN: 3621 (36.2%)
  YELLOW: 4983 (49.8%)
  RED: 1396 (14.0%)

Average Risk Score by Category:
Furniture/Office Chairs    4.0826
Apparel/Menswear           0.7228
Automotive/Brake Pads      0.4209
...
```

### 2. Launch the interactive Plotly dashboard

Make sure you have already run the main simulation to generate `wms_10k_extended_risk_classified.csv`. Then run:

```bash
python "RTHSN Risk based Routing_Plotly V.py"
```

The script will:
- Load the classified CSV.
- Generate four interactive Plotly figures (risk distribution, tier pie, risk by category boxplot, confidence by category boxplot).
- Display a **combined 2×2 dashboard** with all four plots in a single figure.

*Note:* This version uses `plotly` only (no `dash` server). Each figure will open in your browser window individually, and the combined dashboard will also open. You can click, zoom, and hover over data points.

## Customisation

### Changing risk thresholds

Edit the `assign_tier()` function in either script. Default thresholds (from the paper’s recalibrated model) are:
- GREEN ≤ 0.20
- YELLOW ≤ 1.00
- RED > 1.00

### Modifying product categories

Edit the `CATEGORY_INFO` dictionary. Each category requires:
- `hts` (10‑digit HTS code)
- `duty` (ad valorem duty rate, e.g., 0.165 for 16.5%)
- `avg_value` (unit value in EUR/USD)
- `reg_mult` (regulatory penalty multiplier, e.g., 5.0 for pharmaceuticals)
- `scrutiny` (customs scrutiny likelihood, 0.3–1.0)
- `perishable` (True/False)

Also update the `categories_products` dictionary inside `generate_wms_dataset()` with real product description examples.

### Adjusting mock classifier behaviour

In `mock_atlas_classify()`, change:
- `base_conf` values (0.55, 0.85, 0.75) to reflect different LLM confidence patterns.
- Noise standard deviation (currently 0.12).
- Error rate when confidence < 0.65 (currently 30%).

## Output Files Explained

| File | Contents |
|------|----------|
| `wms_inbound_10k_scans_extended.csv` | Raw generated data with product descriptions, category, pallet quantity, lot expiry, declared value, duty rate, regulatory parameters. |
| `wms_10k_extended_risk_classified.csv` | Same as above plus `HTS_Code` (returned by mock classifier), `Confidence`, `Risk_Total`, `Risk_Tier`. |
| `fig_extended_*.png` | Static figures (only from the main script). |

## Dependencies

- Python 3.8+
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `plotly`
- `faker`

(Optional for extended dashboard: `dash`)

## Citation

If you use this code or the RTHSN framework in your research, please cite:

> Pillutla, A. (2026). “A Real‑Time HSN Classification Framework for Warehouse Inbound Operations: Integrating Large Language Models with Material Handling Decision Systems”. Department of Engineering: Logistics and Supply Chain Management, Universitat Autònoma de Barcelona (under review).

## License

This project is provided for academic and research purposes. For commercial use, please contact the author.

## Author

## Acknowledgements

The ATLAS LLM referenced in this framework is developed by FlexifyAI (arXiv:2509.18400). Synthetic data structure is inspired by real‑world WMS schemas.

---

*For issues or questions, please open a GitHub issue or contact the author directly.*
```

Now the equation inside `$$` will render as a proper mathematical expression on GitHub and other platforms that support LaTeX. You can copy and paste this directly into your `README.md` file.
