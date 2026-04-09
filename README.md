# UDHAI — National MSME Analytics Portal

UDHAI is an **AI-powered decision support platform for MSME policy analysis in India**.
It integrates **government MSME datasets, live macroeconomic indicators, machine learning models, and AI-generated insights** into a single interactive dashboard to help policymakers analyze district-level economic patterns and design effective schemes.

The platform provides **geospatial analytics, scheme recommendation, anomaly detection, sentiment analysis, and policy simulation** for India's MSME ecosystem. 

---

# Project Overview

India has **40M+ registered MSMEs**, but policymakers often lack tools to analyze district-level trends and allocate schemes effectively.
UDHAI solves this by creating a **national MSME analytics command center** that combines:

* Official Udyam / UAM enterprise data
* Live macroeconomic indicators
* Machine learning insights
* AI-generated policy analysis
* Interactive geospatial visualization

The system helps identify:

* Regional MSME clusters
* Social inclusion gaps
* Employment generation potential
* Industry diversification
* Policy intervention opportunities

---

# Key Features

## Interactive Dashboard

Five analytical dashboards provide district-level insights:

* Location & Infrastructure Analysis
* Social Inclusion (Gender & Caste Ownership)
* Employment & Investment Analysis
* Industry Sector Mapping
* State Development Scorecard

Each dashboard includes **interactive maps, KPI cards, and analytical charts**.

---

## Decision Support System (DSS)

A policy command center allowing users to:

* Highlight high MSME density states
* Detect low female ownership regions
* Identify high employment generating districts
* View district rankings and insights

---

## AI Scheme Recommendation Engine

The platform automatically recommends **government MSME schemes** based on district profiles.

Examples:

* CGTMSE
* PM MUDRA Yojana
* PM Formalisation of Micro Food Enterprises
* Stand-Up India
* SC/ST Hub Scheme
* RAMP Program

The system analyzes **17 district-level indicators** and ranks the best schemes.

---

## AI District Analyst

Users can ask questions such as:

> “How can female ownership improve in this district?”

The system analyzes district data and **generates AI-powered policy insights using LLaMA 3.2 via Ollama**.

---

## Anomaly Detection (Machine Learning)

UDHAI uses **Isolation Forest** to detect unusual districts based on:

* Women ownership ratio
* SC/ST participation
* Investment per enterprise
* Employment per enterprise

AI then generates a **policy explanation report for these anomalies**.

---

## Live Economic Data Integration

Real-time macro indicators from:

* World Bank API
* IMF Data API
* Udyam Registration Portal
* Google News RSS

This ensures policy recommendations stay **aligned with current economic conditions**.

---

## News Sentiment Analysis

The system automatically:

1. Scrapes district-level business news
2. Uses AI to analyze sentiment
3. Displays a **real-time economic sentiment score**

---

## Policy "What-If" Simulator

Policymakers can simulate interventions such as:

* Financial investment injections
* Increasing female ownership
* Increasing SC/ST entrepreneurship
* Manufacturing sector expansion

The system predicts employment growth and generates an **AI policy feasibility report**.

---

# Technology Stack

### Frontend

* Dash
* Plotly
* Bootstrap

### Backend

* Python
* Flask

### Data Processing

* Pandas
* NumPy

### Machine Learning

* scikit-learn

Models used:

* Isolation Forest (anomaly detection)
* Decision Tree Regressor (future forecasting)

### AI Engine

* Ollama
* LLaMA 3.2

### Data Sources

* data.gov.in
* World Bank API
* IMF DataMapper
* Google News RSS

---

# Project Structure

```
UDHAI/
│
├── app.py
├── live_data.py
├── sentiment_scraper.py
├── data_pipeline.py
├── requirements.txt
│
├── assets/
│   ├── emblem.jpg
│   └── style.css
│
├── JSON/
├── state_wise_csv/
│
├── location_profile.csv
├── social_profile.csv
├── employment_profile.csv
├── industry_profile.csv
├── composite_score.csv
└── msme_merged.csv
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/UDHAI.git
cd UDHAI
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Start Ollama (Required for AI features)

```bash
ollama serve
ollama pull llama3.2
```

## Run Dashboard

```bash
python app.py
```

Open in browser:

```
http://127.0.0.1:8050
```


# Author

**Pravesh Jain**

BTech – Artificial Intelligence & Data Science
Vishwakarma University

Project Focus:
AI + Data Science + GovTech + Policy Analytics

---

# License

This project is intended for **research, academic, and public policy innovation purposes**.
