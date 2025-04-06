# 🧪 Bacterial Growth Analyzer

A modular **Streamlit-based web application** designed for analyzing bacterial growth using optical density (OD) measurements. This tool provides a complete pipeline from file upload and background subtraction to growth model fitting and visualization.

---

## 📌 Features

✅ Upload OD data from 24, 84, 96, or 1536-well plate formats  
✅ Interactive microplate layout with well selection  
✅ Group-based background subtraction  
✅ Group-wise operations (Addition, Subtraction, Multiplication, Division)  
✅ Manual and automatic phase fitting  
✅ Custom ODE model fitting  
✅ Confidence intervals & standard deviation plots  
✅ Exportable plots and results

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
├── app.py                # Main Streamlit app
├── assets/               # Example Excel files and images
├── data/                 # Temporary/user-uploaded data
├── utils/                # Modular helper files
│   ├── layout.py
│   ├── file_io.py
│   ├── background.py
│   ├── operations.py
│   ├── fitting.py
│   ├── models.py
│   ├── plotting.py
│   └── auto_phase.py
├── .gitignore
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 📊 How It Works

### Step-by-Step Workflow

1. **Upload** a CSV file with time and OD values from your microplate reader.
2. **Select layout** (e.g., 96-well).
3. **Pick wells** for blank/background subtraction.
4. **Choose sample wells** and apply subtraction/group operation.
5. **Fit models** to each group or phase:
   - Logistic, Gompertz, Exponential, or Custom ODEs
6. **View plots** with confidence intervals (CI) and standard deviation (SD).
7. **Export** your results or use them for further analysis.

---

## 📷 Screenshots

> Add screenshots here showing:
> - File upload
> - Plate layout
> - Phase fitting
> - CI & SD plots

---

## 🔬 Applications

- Bacterial growth analysis
- Antibiotic resistance studies
- Microbial dynamics research
- Time series OD analysis in experimental biology

---

## 👨‍💻 Technologies Used

- [Streamlit](https://streamlit.io/)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [SciPy](https://scipy.org/)
- [Plotly](https://plotly.com/python/)

---

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 🌐 Author

**Srinath Laka**  
Master’s in Scientific Instrumentation | Ernst Abbe Hochschule (EAH) Jena  
📧 Email: *optional*  
🌍 Project: [GitHub Repo Link](https://github.com/srinathlaka/bacterial-growth-analyzer)
