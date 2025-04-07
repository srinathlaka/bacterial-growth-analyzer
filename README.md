# 🧪 Bacterial Growth Analyzer

A modular **Streamlit-based web application** designed for analyzing bacterial growth using optical density (OD) measurements. 
This tool provides a complete pipeline from file upload and background subtraction to growth model fitting and visualization, 
with both manual and automatic phase detection methods.

---

## 📌 Features

✅ Upload OD data from 24, 84, 96, or 1536-well plate formats  
✅ Interactive microplate layout with well selection  
✅ Group-based background subtraction (supports multiple groups)  
✅ Group-wise operations (Addition, Subtraction, Multiplication, Division)  
✅ Manual and automatic phase detection (threshold + slope methods)  
✅ Nonlinear curve fitting with Logistic, Gompertz, and Exponential models  
✅ Custom ODE solver integration possible  
✅ Confidence intervals & standard deviation plots  
✅ Exportable plots and results (CSV/Excel format)

---

## 🚀 Getting Started

### 📦 Prerequisites

- Python 3.8+
- pip (Python package manager)

### 🔧 Installation

Clone the repository:
```bash
git clone https://github.com/srinathlaka/bacterial-growth-analyzer.git
cd bacterial-growth-analyzer
```

Install the required packages:
```bash
pip install -r requirements.txt
```

Run the app:
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
bacterial-growth-analyzer/
├── app.py                  # Main Streamlit app (UI and workflow)
├── assets/                 # Example Excel files and layout images
├── data/                   # Temporary/user-uploaded data
├── utils/                  # Modular logic split into utility files
│   ├── file_io.py          # File parsing and layout selection
│   ├── layout.py           # UI layout for button grid of wells
│   ├── background.py       # Background subtraction logic
│   ├── operations.py       # Math operations across groups
│   ├── fitting.py          # Curve fitting and parameter setup
│   ├── models.py           # Growth models: logistic, gompertz, exponential
│   ├── plotting.py         # Plotly-based visualizations
│   └── auto_phase.py       # Phase detection and CI overlays
├── .gitignore
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 📊 How It Works

### Step-by-Step Workflow

1. **Upload** your OD data in `.csv` or `.xlsx` format.
2. **Select plate layout** and preview time series.
3. **Choose blank and sample wells** for each group.
4. **Subtract background** within groups.
5. **Apply group operations** (if needed).
6. **Fit models** per group or well (manual phase selection or automatic detection).
7. **View results with CI/SD**, export plots or table of parameters.

---

## 📷 Screenshots

> _Coming soon_ — visualize layout selection, phase highlighting, and fit diagnostics.

---

## 🔬 Applications

- Bacterial growth curve analysis
- Antibiotic resistance and inhibition zone profiling
- Biotech assay optimization
- Experimental biology OD/time-series modeling

---

## 👨‍💻 Technologies Used

- [Streamlit](https://streamlit.io/)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [SciPy](https://scipy.org/)
- [Plotly](https://plotly.com/python/)
- [Ruptures](https://centre-borelli.github.io/ruptures-docs/) (for change point detection)

---

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to propose or add.

---

## 🌐 Author

**Srinath Laka**  
Master’s in Scientific Instrumentation | Ernst Abbe Hochschule (EAH) Jena  
🌍 GitHub: [srinathlaka](https://github.com/srinathlaka)  
📧 Email: srinathlaka1@gmail.com
