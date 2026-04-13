# ⚕️ Is Your Medical AI Fair? - Healthcare AI Fairness Audit Platform

A comprehensive, research-backed tool for detecting demographic bias in healthcare AI models **before deployment**. Designed specifically for diverse urban healthcare populations.

**[🔗 Live Demo](https://healthcare-ai-fairness-audit.streamlit.app)** | **[📧 Contact](mailto:gaurloveleen@yahoo.com)** | **[ORCID](https://orcid.org/0000-0002-0885-1550)**

---

## 🎯 Overview

### The Problem
Medical AI models trained on biased data often work differently for different demographic groups:
- 95% accurate for English speakers, 62% for Spanish speakers
- 88% accurate for privately insured, 52% for uninsured patients
- 85% accurate for White patients, 45% for Hispanic patients

**This hidden bias harms vulnerable populations.** Standard accuracy metrics miss it entirely.

### The Solution
This platform provides an **interactive, scientific approach** to detect demographic bias in healthcare AI **before it harms patients**. 

Written in plain English. No technical background required. Results in 5 minutes.

### Key Features

- 📊 **Interactive Demo** - Realistic synthetic data showing real healthcare disparities
- 👥 **Multiple Demographics** - Check fairness by ethnicity, language, insurance type, age, or any demographic
- 📤 **Upload Your Data** - Audit your own model predictions
- 📈 **Three Fairness Checks:**
  - ✅ **Equal Accuracy** - Does AI work equally for all groups?
  - ✅ **Equal Diagnosis** - Does AI diagnose disease at the same rate?
  - ✅ **Equal Errors** - Do some groups get more false alarms?
- 🎨 **Clear Visualizations** - Color-coded results (✅ safe, ⚠️ needs fixing, 🛑 do not deploy)
- ⚡ **Instant Results** - Get insights in seconds
- 📋 **Deployment Recommendations** - Clear guidance on whether to use the AI

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

### The Demo (Realistic Diverse Population)

The demo includes **300 realistic patients** reflecting a diverse urban healthcare setting:
- 70% Hispanic/Latino
- 15% African American
- 15% White/Other

**With realistic disparities built in:**
- Spanish speakers: 38% diagnostic accuracy ❌
- English speakers: 87% diagnostic accuracy ✅
- Uninsured patients: 52% accuracy ❌
- Private insurance: 88% accuracy ✅

**When you use the demo, you'll discover these disparities yourself.**

### Three Fairness Checks

The platform tests **three critical fairness metrics:**

#### ✅ Check 1: Equal Accuracy
**Question:** Does the AI work equally well for all groups?

```
Example:
  Hispanic patients: 45% accuracy
  White patients: 85% accuracy
  Disparity: 40% ❌ HUGE! (Threshold: 10%)
```

#### ✅ Check 2: Equal Diagnosis Rate  
**Question:** Does AI diagnose disease at the same rate for all groups?

```
Example:
  Spanish speakers: 32% diagnosed
  English speakers: 56% diagnosed
  Disparity: 24% ❌ (Threshold: 10%)
```

#### ✅ Check 3: Equal Error Rates
**Question:** Do some groups get more false alarms than others?

```
Example:
  Uninsured: 15% false alarm rate
  Private insurance: 5% false alarm rate
  Disparity: 10% ⚠️ (Threshold: 10%)
```

### Deployment Recommendation

Based on results:
- ✅ **APPROVED** - All metrics pass → Safe to deploy
- ⚠️ **NEEDS FIXING** - Some disparity → Fix before use
- 🛑 **DO NOT DEPLOY** - Critical bias → Major problems

---

## 📥 Usage

### Demo Tab - Try It Yourself

1. Click **"📊 See Demo"** tab
2. Review 300 realistic patient scenarios
3. **Select demographic to check:**
   - Age group (45-55, 56-65, 66+ years)
   - Ethnicity (Hispanic/Latino, African American, White/Other)
   - Primary language (Spanish, English/Bilingual, English)
   - Insurance type (Private, Medicare, Medicaid, Uninsured)
4. Adjust fairness threshold (default: 10%)
5. **View instant fairness analysis**
   - See results for each demographic group
   - Identify disparities
   - Get deployment recommendation

### Upload Your Own Data

1. Prepare CSV with columns: `prediction`, `ground_truth`, plus any demographic column
2. Upload file
3. Select which demographic(s) to analyze
4. Run analysis
5. View your model's fairness report

### Data Format Example

```csv
prediction,ground_truth,age_group,primary_language,insurance_type
1,1,56-65,Spanish,Medicaid
0,0,66+,English,Medicare
1,1,45-55,English,Private
0,1,56-65,Spanish,Uninsured
```

**Columns:**
- `prediction` - AI's prediction (0 = no disease, 1 = disease)
- `ground_truth` - Actual outcome (0 = no disease, 1 = disease)
- `age_group` - Patient age group (optional)
- `primary_language` - Patient's language (optional)
- `insurance_type` - Insurance coverage (optional)
- Any other demographic you want to check

---

## 📊 Understanding Fairness Metrics

### Metric 1: Equal Accuracy
**What it measures:** Does AI work the same for all groups?

Healthcare example:
```
Group A (English speakers): 87% accurate
Group B (Spanish speakers): 62% accurate
Disparity: 25% ❌ (Threshold: 10%)
→ AI works much worse for Spanish speakers!
```

### Metric 2: Equal Diagnosis Rate  
**What it measures:** Does AI diagnose disease at the same rate?

Healthcare example:
```
Group A (Private insurance): 56% diagnosed
Group B (Uninsured): 32% diagnosed  
Disparity: 24% ❌ (Threshold: 10%)
→ AI misses disease in uninsured patients!
```

### Metric 3: Equal Error Rates
**What it measures:** Do some groups get more false alarms?

Healthcare example:
```
Group A (African American): 15% false alarm rate
Group B (White): 5% false alarm rate
Disparity: 10% ⚠️ (Threshold: 10%)
→ African American patients get unnecessary alarms!
```

### Why These Three Matter

1. **Accuracy** → If it's wrong for your group, diagnoses fail
2. **Diagnosis Rate** → If AI diagnoses less, diseases get missed
3. **Error Rates** → If it alarms more for your group, you get anxious & waste resources

**All three must pass** for AI to be fair to everyone! ✅

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

## 📈 Current Features ✅

- [x] Interactive demo with realistic diverse population data
- [x] Multiple demographic analysis (ethnicity, language, insurance, age)
- [x] Three core fairness metrics (Accuracy, Diagnosis Rate, Error Rates)
- [x] CSV data upload for custom models
- [x] Plain-English explanations (no jargon)
- [x] Color-coded results (safe/warning/fail)
- [x] Deployment recommendations
- [x] Performance optimized for instant results
- [x] Mobile-friendly interface
- [x] Research-backed methodology

## 🗓️ Planned Features

- [ ] Confidence intervals and statistical significance
- [ ] PDF report export
- [ ] Intersectional analysis (combinations of demographics)
- [ ] Confusion matrices by demographic group
- [ ] ROC curves and threshold optimization
- [ ] Model comparison tool
- [ ] EHR system integration
- [ ] Real-time monitoring dashboard
- [ ] Community feedback dashboard
- [ ] Spanish-language interface

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

## 🏥 Health Equity Focus

This tool is specifically designed to address health disparities:

### Why It Matters
- **Hidden Bias Harms Patients** - AI bias affects real diagnoses
- **Vulnerable Populations at Risk** - Minorities, uninsured, non-English speakers suffer most
- **Legal Requirements** - Fair AI is now legally required in many jurisdictions
- **Ethical Obligation** - Healthcare should serve EVERYONE equally

### Built-In Disparities Detection
The demo and analysis specifically check for:
- **Language barriers** - Spanish vs English speaking patients
- **Economic disparities** - Insured vs uninsured patients
- **Racial/Ethnic differences** - AI outcomes across demographics
- **Age disparities** - Elderly vs younger patients
- **Intersectional effects** - Combined demographic factors

### Research Foundation
Built on peer-reviewed fairness research and health equity principles. All metrics validated in healthcare AI literature.

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

### Primary Focus: Healthcare ⚕️
- **Diagnosis AI** - Ensure cancer detection works for all races
- **Triage Systems** - Fair emergency room prioritization
- **Risk Prediction** - Equal sepsis/mortality risk assessment
- **Treatment Planning** - Fair clinical decision support
- **Health Screening** - Equitable disease detection

### Before Deployment Checklist
- [ ] Test AI with this platform
- [ ] Check all demographic groups
- [ ] Ensure all metrics pass
- [ ] Document fairness testing
- [ ] Get ethics approval
- [ ] Deploy with confidence

---

**Made with ❤️ by Dr. Loveleen Gaur**

For AI fairness and health equity. 🏥

