# 📚 Book Recommendation System

An intelligent, interactive web application built with Streamlit that helps users discover their next favorite book.  
This system uses machine learning-based recommendation techniques to provide personalized suggestions, search functionality, and insights into large-scale book data.

---

## 🚀 Features

### 📊 Comprehensive Dashboard (Home)
- Displays real-time dataset statistics:
  - 271,360+ books  
  - 278,858+ users  
  - 1.1M+ ratings  

---

### 🔍 Intelligent Search
- Search books by:
  - Title  
  - Author  
  - ISBN  

---

### 🎯 Personalized Recommendations
- Machine learning-based recommendation engine  
- Uses cosine similarity on book titles  
- Suggests books based on user preferences  

---

### 📖 Popular Books Discovery
- “Popular Books” section with rating filters  
- Interactive slider for top-rated books  
- Highlights trending titles  

---

### 🎨 Clean UI/UX
- Sidebar navigation system  
- Simple and modern interface  
- Smooth user experience  

---

## 🛠️ Tech Stack

- Language: Python  
- Framework: Streamlit  
- Data Processing: Pandas, NumPy  
- Machine Learning: Scikit-learn (Cosine Similarity)  
- Deployment: AWS EC2  

---

## ⚙️ How to Run the Project Locally

### 1. Prerequisites
- Download dataset from Kaggle
- Python 3.8 or higher installed  

---

### 2. Clone the Repository
```bash
git clone https://github.com/your-username/book-recommendation-system.git
cd book-recommendation-system

---

### 3. Create Virtual Environment (Recommended)
```bash
python -m venv .venv

### 4. Activate Virtual Environment
```bash
.venv\Scripts\activate

---
### 5. Install Dependencies
```bash
pip install -r requirements.txt

---
### 6. Run the Application
```bash
streamlit run app.py
