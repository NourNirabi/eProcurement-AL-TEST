# **eProcurement-AL-TEST: Automated Product Classification & Scraping System**  

🚀 **A powerful pipeline for scraping, processing, and classifying construction products using AI (LLMs + Embeddings) and web scraping.**  

---

## **🔍 Overview**  
This project automates:  
1. **Web Scraping**: Extracts product data (e.g., Sika GCC) with `sika_scraper.py`.  
2. **AI Classification**: Uses `classifier_emb.py` (high-accuracy embeddings) and `classifier_llm.py` (LLM-based) to categorize products into a 4-level hierarchy.  
3. **Data Pipeline**: Processes raw JSON/CSV into classified outputs (`classified_products.csv`).  

---

## **⚙️ Features**  
| Component | Key Functionality |  
|-----------|------------------|  
| **`sika_scraper.py`** | Scrapes product names, specs, PDFs, and technical data from Sika GCC. |  
| **`classifier_emb.py`** | Embeds-based classifier (fast & accurate). Updates taxonomy dynamically. |  
| **`classifier_llm.py`** | LLM-powered classifier (Ollama) for complex cases. |  
| **`classification_tree.csv`** | Defines the hierarchical taxonomy (Group → Category → Class → Type). |  

---

## **🚀 Quick Start**  
### **1. Install Dependencies**  
```bash  
pip install -r requirements.txt  # requests, bs4, pdfplumber, ollama, pandas  
```  

### **2. Run the Scraper**  
```bash  
python sites/sika_scraper.py  # Output: data/sika_product_data.json  
```  

### **3. Classify Products**  
```bash  
python classification/classifier_emb.py  # Uses data/sika_product_data.json → classified_products.csv  
```  

---

## **📂 Project Structure**  
```  
eProcurement-AL-TEST/  
├── classification/          # AI classifiers & taxonomy  
├── data/                   # Raw/processed JSON, CSV, links  
├── sites/                  # Web scrapers (e.g., Sika)  
├── requirements.txt        # Python dependencies  
└── README.md               # This file  
```  

---

## **🛠️ Customization**  
- **Taxonomy**: Edit `classification_tree.csv` to modify categories.  
- **Scraping**: Adapt `sika_scraper.py` for other sites (e.g., edit `BASE_URL`).  
- **AI Models**: Swap Ollama for OpenAI/Gemini in `classifier_llm.py`.  

---

## **📈 Performance**  
| Classifier | Accuracy | Speed | Use Case |  
|------------|---------|-------|----------|  
| `classifier_emb.py` | ✅ 95%+ | ⚡ Fast | Production |  
| `classifier_llm.py` | ⚠️ ~70% | 🐌 Slow | Experimental |  

---
--- 

**Let’s build smarter procurement!** 🔨🤖
