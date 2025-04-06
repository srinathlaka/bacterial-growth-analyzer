# Bacterial Growth Analysis App

A modular Streamlit web application for analyzing optical density (OD) data in biological research. Supports:
- Background subtraction
- Phase-based model fitting
- Custom ODE modeling
- Confidence interval & standard deviation visualization

## 🔧 Features

- Upload 96/84/24/1536-well plate CSV files
- Interactive well selection grid
- Group-wise operations (Add/Subtract/etc.)
- Manual & automatic phase fitting
- Custom ODE-based modeling
- Clean UI and modular backend

## 📁 Project Structure

- `app.py` — Main Streamlit app
- `assets/` — Example files & background images
- `utils/` — Modular codebase (background, fitting, layout, etc.)

## ▶️ Run the App

```bash
streamlit run app.py
```

## 📦 Dependencies

Install using:

```bash
pip install -r requirements.txt
```

## 🧪 Example Files

You can find example CSV files and layout in the `assets/` folder.
