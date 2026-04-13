# ⚕️ AI Fairness Audit Platform

A comprehensive, research-backed tool for detecting demographic bias in AI/ML models used in healthcare and other high-stakes applications.

**[🔗 Live Demo](https://healthcare-ai-fairness-audit.streamlit.app)** | **[📧 Contact](mailto:gaurloveleen@yahoo.com)**

---

## 🎯 Overview

Medical AI models often exhibit hidden bias that affects different demographic groups differently. This platform provides an interactive, scientific approach to **detect**, **measure**, and **report** on demographic bias before deployment.

### Key Features

- 📊 **Interactive Demo** - Pre-loaded example with synthetic healthcare AI data
- 📤 **Custom Data Upload** - Audit your own model predictions
- 📈 **Multiple Fairness Metrics** - Demographic Parity, Equalized Odds, Accuracy Parity, Equal Opportunity
- 🎨 **Beautiful Visualizations** - Clear, actionable charts and tables
- ⚡ **Fast Performance** - Optimized for instant results
- 📋 **Professional Reports** - Detailed fairness analysis with deployment recommendations

---

## 🚀 Quick Start

### Option 1: Try Live (Recommended)

Open the live app: **https://healthcare-ai-fairness-audit.streamlit.app**

### Option 2: Run Locally

```bash
# Clone repository
git clone https://github.com/YOUR-USERNAME/healthcare-ai-fairness-audit.git
cd healthcare-ai-fairness-audit

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Option 3: Deploy Your Own Instance

1. Fork this repository
2. Go to https://share.streamlit.io
3. Create new app and point to your repo
4. Deploy in seconds

---

## 📊 How It Works

### The Problem

Standard AI model accuracy metrics hide bias:
```
Overall accuracy: 95%
Group A accuracy: 98%
Group B accuracy: 92%
Disparity: 6% (hidden!)
```

This tool reveals these hidden disparities.

### The Solution

The platform computes **four fairness metrics** across demographic groups:

| Metric | Definition | Threshold |
|--------|-----------|-----------|
| **Demographic Parity** | Equal positive prediction rates | < 10% difference |
| **Equalized Odds** | Equal false positive rates | < 10% difference |
| **Accuracy Parity** | Consistent accuracy | < 10% difference |
| **Equal Opportunity** | Equal true positive rates | < 10% difference |

### Deployment Recommendation

- ✅ **APPROVED** - All metrics pass
- ⚠️ **CONDITIONAL** - Address fairness gaps
- 🛑 **DO NOT DEPLOY** - Critical concerns

---

## 📥 Usage

### Demo Tab

1. Click **"📊 Demo"** tab
2. Select demographic attribute (age_group or gender)
3. Adjust fairness threshold slider
4. View instant fairness analysis

### Upload Tab

1. Prepare CSV with: `prediction`, `ground_truth`, `demographic_column`
2. Upload file
3. Select column mappings
4. Run analysis

### Data Format Example

```csv
prediction,ground_truth,age_group,gender
1,1,55-65,M
0,0,65-75,F
1,1,75+,M
```

---

## 📊 Fairness Metrics Explained

### Demographic Parity
Do all groups get positive predictions at equally?
```
Group A: 65% positive rate
Group B: 35% positive rate
Disparity: 30% ❌
```

### Equalized Odds
Are false positive rates similar across groups?
```
Group A FPR: 15%
Group B FPR: 30%
Disparity: 15% ❌
```

### Accuracy Parity
Is model accuracy consistent?
```
Group A: 92% accuracy
Group B: 72% accuracy
Disparity: 20% ❌
```

### Equal Opportunity
Do groups have similar true positive rates?
```
Group A TPR: 88%
Group B TPR: 65%
Disparity: 23% ❌
```

---

## 🔧 Installation

### Requirements
- Python 3.8+
- pip or conda

### Setup

```bash
git clone https://github.com/YOUR-USERNAME/healthcare-ai-fairness-audit.git
cd healthcare-ai-fairness-audit

# Optional: Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run
streamlit run app.py
```

---

## 📈 Features

### Current ✅
- [x] Interactive demo
- [x] CSV upload
- [x] 4 fairness metrics
- [x] Beautiful visualizations
- [x] Deployment recommendations
- [x] Performance optimized

### Roadmap 🗓️
- [ ] Confidence intervals
- [ ] Statistical significance tests
- [ ] Export to PDF/CSV
- [ ] Confusion matrices
- [ ] ROC curves
- [ ] Threshold optimization
- [ ] Model comparison
- [ ] Intersectional analysis

---

## 🎓 Research Foundation

**Author:** Dr. Loveleen Gaur

**Credentials:**
- USCIS EB1A Extraordinary Ability in AI
- Elsevier-Stanford Top 2% Scientists (2024-2025)
- 130+ peer-reviewed publications
- 30+ published books on AI ethics

**Published Research:**
- Gaur, L., et al. (2023). "HCI-driven XAI model for disease detection."
  *ACM Transactions on Multimedia Computing, Communications, and Applications*.

**Contact:**
- Email: gaurloveleen@yahoo.com
- ORCID: 0000-0002-0885-1550

---

## 📚 Key References

1. **Fairness Definitions:** Barocas, S., Hardt, M., & Narayanan, A. (2019)
2. **Demographic Parity:** Calmon, F., et al. (2017) 
3. **Equalized Odds:** Hardt, M., Price, E., & Srebro, N. (2016)
4. **Explainable AI:** Selvaraju, R. R., et al. (2017)
5. **Healthcare AI:** Rajkomar, A., et al. (2018)

---

## 🔒 Privacy & Security

- ✅ **No Data Storage** - All processing in-memory
- ✅ **No Tracking** - Complete user privacy
- ✅ **Open Source** - Publicly auditable code
- ✅ **Academic Foundation** - Peer-reviewed methodology

---

## 🚀 Deployment

### Live Instance
https://healthcare-ai-fairness-audit.streamlit.app

### Local
```bash
streamlit run app.py
```

### Streamlit Cloud
1. Fork repository
2. Go to https://share.streamlit.io
3. Connect GitHub
4. Deploy

---

## 🤝 Contributing

Contributions welcome! 

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push branch (`git push origin feature/amazing`)
5. Open Pull Request

---

## 📋 License

MIT License - See LICENSE file for details

---

## ❓ FAQ

**Q: Is my data safe?**  
A: Yes. Data is never stored; all processing is in-memory.

**Q: Can I use for production?**  
A: Yes, designed for pre-deployment auditing.

**Q: Minimum sample size?**  
A: At least 30 per group recommended.

**Q: How to interpret thresholds?**  
A: Default 10% - adjust based on your needs.

---

## 📞 Support

- **Issues:** GitHub Issues
- **Email:** gaurloveleen@yahoo.com
- **ORCID:** 0000-0002-0885-1550

---

## 🎯 Use Cases

- **Healthcare:** Pre-deployment AI audit
- **Finance:** Lending and credit systems
- **Criminal Justice:** Risk assessment
- **Employment:** Hiring decisions

---

**Made with ❤️ by Dr. Loveleen Gaur**

For AI fairness and health equity. 🏥

