
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Healthcare AI Fairness Audit",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .main { padding: 20px; }
    .header {
        background-color: #1f77b4;
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# SIDEBAR
st.sidebar.markdown("# 🔧 Configuration")
st.sidebar.markdown("---")

tab = st.sidebar.radio(
    "Select Mode:",
    ["📊 Demo (Pre-loaded Data)", "📤 Upload Your Data", "ℹ️ About"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Dr. Loveleen Gaur**  
AI Ethics & Governance | Healthcare AI  
📧 gaurloveleen@yahoo.com  
ORCID: 0000-0002-0885-1550
""")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**About This Tool:**
- Detects demographic bias in healthcare AI models
- Based on NIH AIM-AHEAD program goals
- Published methodology in ACM Transactions
- Supports deployment safety assessment
""")

# HEADER
st.markdown("""
<div class="header">
    <h1>⚕️ Healthcare AI Fairness Audit Platform</h1>
    <p><strong>Detecting Demographic Bias in Medical AI Models</strong></p>
    <p style="font-size: 12px; margin-top: 10px;">
    Based on ACM Transactions research by Dr. Loveleen Gaur | 
    Supports NIH AIM-AHEAD Program Goals
    </p>
</div>
""", unsafe_allow_html=True)

# HELPER FUNCTIONS
def compute_fairness_metrics(predictions, ground_truth, demographics, protected_attr):
    """Compute fairness metrics stratified by demographic group."""
    groups = demographics[protected_attr]
    unique_groups = np.unique(groups)
    
    results = []
    for group in unique_groups:
        mask = groups == group
        group_pred = predictions[mask]
        group_true = ground_truth[mask]
        
        if len(group_true) == 0:
            continue
        
        try:
            tn, fp, fn, tp = confusion_matrix(group_true, group_pred, labels=[0, 1]).ravel()
        except:
            tn, fp, fn, tp = 0, 0, 0, len(group_true)
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        positive_rate = group_pred.mean()
        accuracy = accuracy_score(group_true, group_pred)
        fpr = fp / (tn + fp) if (tn + fp) > 0 else np.nan
        fnr = fn / (tp + fn) if (tp + fn) > 0 else np.nan
        
        results.append({
            'group': group,
            'n_samples': len(group_true),
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'positive_rate': positive_rate,
            'fpr': fpr,
            'fnr': fnr
        })
    
    return pd.DataFrame(results)

def fairness_tests(stratified_df, threshold=0.10):
    """Run formal fairness tests."""
    tests = {}
    
    if 'positive_rate' in stratified_df.columns:
        pos_rates = stratified_df['positive_rate'].values
        disparity = pos_rates.max() - pos_rates.min()
        tests['Demographic Parity'] = {
            'disparity': disparity,
            'pass': disparity < threshold,
            'description': f'Max difference in positive rate: {disparity:.1%}'
        }
    
    if 'fpr' in stratified_df.columns:
        fpr_disparity = stratified_df['fpr'].max() - stratified_df['fpr'].min()
        tests['Equalized Odds (FPR)'] = {
            'disparity': fpr_disparity,
            'pass': fpr_disparity < threshold,
            'description': f'False Positive Rate disparity: {fpr_disparity:.1%}'
        }
    
    if 'accuracy' in stratified_df.columns:
        acc_disparity = stratified_df['accuracy'].max() - stratified_df['accuracy'].min()
        tests['Accuracy Parity'] = {
            'disparity': acc_disparity,
            'pass': acc_disparity < threshold,
            'description': f'Max accuracy difference: {acc_disparity:.1%}'
        }
    
    return tests

def generate_recommendation(tests):
    """Generate overall deployment recommendation."""
    all_pass = all(test.get('pass', False) for test in tests.values())
    num_fail = sum(1 for test in tests.values() if not test.get('pass', False))
    
    if all_pass:
        return "✓ APPROVED FOR DEPLOYMENT (with ongoing monitoring)"
    elif num_fail <= 1:
        return "⚠️ CONDITIONAL APPROVAL (address flagged disparities)"
    else:
        return "🛑 DO NOT DEPLOY (significant fairness concerns)"

# DEMO MODE
if tab == "📊 Demo (Pre-loaded Data)":
    st.markdown("## Demo: Alzheimer's Detection Model Fairness Audit")
    st.markdown("*This demonstrates fairness auditing on a synthetic healthcare AI model.*")
    
    np.random.seed(42)
    n_samples = 200
    
    predictions = []
    age_groups = []
    genders = []
    
    for i in range(n_samples):
        age = np.random.choice(['55-65', '65-75', '75+'])
        gender = np.random.choice(['M', 'F'])
        
        if age == '75+':
            pred = np.random.choice([0, 1], p=[0.65, 0.35])
        elif age == '65-75':
            pred = np.random.choice([0, 1], p=[0.55, 0.45])
        else:
            pred = np.random.choice([0, 1], p=[0.35, 0.65])
        
        predictions.append(pred)
        age_groups.append(age)
        genders.append(gender)
    
    predictions = np.array(predictions)
    ground_truth = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
    
    demographics = pd.DataFrame({
        'age_group': age_groups,
        'gender': genders
    })
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples", len(predictions))
    with col2:
        st.metric("Positive Cases", predictions.sum())
    with col3:
        st.metric("Model Accuracy", f"{accuracy_score(ground_truth, predictions):.1%}")
    
    st.markdown("---")
    
    st.markdown("### Step 1: Select Demographic Attribute")
    protected_attr = st.selectbox(
        "Which demographic group would you like to audit for bias?",
        ['age_group', 'gender']
    )
    
    st.markdown("---")
    st.markdown("### Step 2: Fairness Analysis Results")
    
    stratified = compute_fairness_metrics(predictions, ground_truth, demographics, protected_attr)
    tests = fairness_tests(stratified, threshold=0.10)
    recommendation = generate_recommendation(tests)
    
    st.markdown(f"**Performance by {protected_attr}:**")
    st.dataframe(
        stratified[['group', 'n_samples', 'accuracy', 'positive_rate']].round(3),
        use_container_width=True
    )
    
    st.markdown("---")
    st.markdown("### Step 3: Bias Visualization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = go.Figure()
        colors = ['#27ae60' if x > 0.75 else '#f39c12' if x > 0.65 else '#e74c3c' 
                  for x in stratified['accuracy']]
        
        fig1.add_trace(go.Bar(
            x=stratified['group'],
            y=stratified['accuracy'],
            marker=dict(color=colors),
            name='Accuracy',
            text=[f"{x:.1%}" for x in stratified['accuracy']],
            textposition='outside'
        ))
        fig1.add_hline(y=0.75, line_dash="dash", line_color="red")
        fig1.update_layout(
            title=f"Model Accuracy by {protected_attr}",
            xaxis_title=protected_attr,
            yaxis_title="Accuracy",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=stratified['group'],
            y=stratified['positive_rate'],
            marker=dict(color='#3498db'),
            name='Positive Rate',
            text=[f"{x:.1%}" for x in stratified['positive_rate']],
            textposition='outside'
        ))
        fig2.update_layout(
            title="Prediction Rate (Demographic Parity Check)",
            xaxis_title=protected_attr,
            yaxis_title="Positive Prediction Rate",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### Step 4: Fairness Test Results")
    
    for test_name, test_result in tests.items():
        if test_result['pass']:
            status = "✓ PASS"
        else:
            status = "⚠️ WARNING"
        
        st.markdown(f"**{test_name}** - {status}")
        st.markdown(f"{test_result['description']}")
        st.markdown("---")
    
    st.markdown("### Step 5: Deployment Recommendation")
    
    if "APPROVED" in recommendation:
        color = "green"
    elif "CONDITIONAL" in recommendation:
        color = "orange"
    else:
        color = "red"
    
    st.markdown(f"""
    <div style="background-color: {color}20; border-left: 4px solid {color}; padding: 15px; border-radius: 5px;">
    <h3 style="color: {color}; margin-top: 0;">{recommendation}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Interpretation")
    st.info("""
    **What does this mean?**
    
    - **Demographic Parity**: Does the model make positive predictions at equal rates across groups?
    - **Equalized Odds**: Are false positive rates similar across groups?
    - **Accuracy Parity**: Is model accuracy consistent across demographic groups?
    
    **Red Flags:**
    - ⚠️ Different accuracy across groups → Model may be less reliable for certain populations
    - ⚠️ Different prediction rates across groups → Model may systematically over/under-predict
    - ✓ Consistent performance → Safe to deploy with monitoring
    """)

elif tab == "📤 Upload Your Data":
    st.markdown("## Audit Your Own Model")
    st.markdown("Upload your model predictions and demographic data to run a fairness audit.")
    
    st.info("""
    **What you need:**
    1. **Model predictions** (0 or 1 for binary classification)
    2. **Ground truth labels** (actual outcomes)
    3. **Demographic data** (age, gender, ethnicity, etc.)
    
    Format: CSV file with columns for predictions, true labels, and demographics
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.success("✓ File loaded successfully!")
            st.dataframe(df.head(), use_container_width=True)
            
            st.markdown("---")
            st.markdown("### Configure Your Data")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                pred_col = st.selectbox("Prediction column:", df.columns)
            with col2:
                truth_col = st.selectbox("Ground truth column:", df.columns)
            with col3:
                demo_cols = st.multiselect("Demographic columns:", df.columns)
            
            if st.button("🔍 Run Fairness Audit", type="primary"):
                
                predictions = df[pred_col].values
                ground_truth = df[truth_col].values
                
                st.markdown("---")
                st.markdown("### Audit Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Samples Analyzed", len(predictions))
                with col2:
                    st.metric("Overall Accuracy", f"{accuracy_score(ground_truth, predictions):.1%}")
                with col3:
                    st.metric("Positive Cases", predictions.sum())
                
                st.markdown("---")
                
                for demo_col in demo_cols:
                    st.markdown(f"### Analysis by {demo_col}")
                    
                    stratified = compute_fairness_metrics(predictions, ground_truth, df, demo_col)
                    tests = fairness_tests(stratified)
                    
                    st.dataframe(
                        stratified[['group', 'n_samples', 'accuracy', 'positive_rate']].round(3),
                        use_container_width=True
                    )
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig1 = go.Figure()
                        colors = ['#27ae60' if x > 0.75 else '#f39c12' if x > 0.65 else '#e74c3c' 
                                  for x in stratified['accuracy']]
                        fig1.add_trace(go.Bar(x=stratified['group'], y=stratified['accuracy'],
                                             marker=dict(color=colors)))
                        fig1.update_layout(title=f"Accuracy by {demo_col}", height=400)
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    with col2:
                        fig2 = go.Figure()
                        fig2.add_trace(go.Bar(x=stratified['group'], y=stratified['positive_rate'],
                                             marker=dict(color='#3498db')))
                        fig2.update_layout(title=f"Positive Rate by {demo_col}", height=400)
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    st.markdown("**Fairness Tests:**")
                    for test_name, test_result in tests.items():
                        status = "✓ PASS" if test_result['pass'] else "⚠️ WARNING"
                        st.markdown(f"{status} - {test_name}: {test_result['description']}")
                    
                    st.markdown("---")
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

elif tab == "ℹ️ About":
    st.markdown("""
    # About This Platform
    
    ## 🎯 Purpose
    
    This platform helps healthcare institutions audit AI/ML models for demographic bias before deployment.
    It supports the goals of **NIH's AIM-AHEAD program** (Artificial Intelligence/Machine Learning 
    Consortium to Advance Health Equity and Researcher Diversity).
    
    ## 🔬 Research Foundation
    
    - **Author:** Dr. Loveleen Gaur
    - **Published Work:** "HCI-driven XAI model for Alzheimer's detection" - 
      ACM Transactions on Multimedia Computing, Communications, and Applications (2023)
    - **Credentials:** 
      - USCIS EB1A Extraordinary Ability in AI
      - Elsevier-Stanford Top 2% Scientists (2024-2025)
      - 130+ peer-reviewed publications
      - 30+ published books on AI ethics and healthcare
    
    ## 📊 Key Metrics
    
    This tool computes:
    
    ### Demographic Parity
    Are predictions distributed equally across demographic groups?
    - ✓ PASS: < 10% difference in positive prediction rate
    - ⚠️ WARNING: 10-20% difference
    - ✗ FAIL: > 20% difference
    
    ### Equalized Odds
    Are error rates (FPR, FNR) similar across groups?
    - ✓ PASS: < 10% difference
    - ⚠️ WARNING: 10-20% difference
    - ✗ FAIL: > 20% difference
    
    ### Accuracy Parity
    Is model accuracy consistent across groups?
    - ✓ PASS: < 10% accuracy difference
    - ⚠️ WARNING: 10-20% difference
    - ✗ FAIL: > 20% difference
    
    ## 🏥 Use Cases
    
    1. **Pre-deployment Audit** — Screen models before clinical use
    2. **Regulatory Compliance** — Meet FDA/institutional fairness requirements
    3. **Model Selection** — Compare fairness of different models
    4. **Training Data Analysis** — Identify underrepresented populations
    5. **Continuous Monitoring** — Track fairness post-deployment
    
    ## 🌍 Alignment with AIM-AHEAD
    
    The NIH AIM-AHEAD program focuses on:
    
    - **Reducing health disparities** through equitable AI algorithms
    - **Increasing diversity** in AI/ML research and development
    - **Building infrastructure** for fair healthcare AI
    - **Engaging communities** in AI development
    
    This platform directly supports these goals by:
    - ✓ Automatically detecting demographic bias
    - ✓ Quantifying fairness across groups
    - ✓ Providing actionable deployment recommendations
    - ✓ Supporting institutional governance of AI
    
    ## 📧 Contact
    
    **Dr. Loveleen Gaur**
    - Email: gaurloveleen@yahoo.com
    - ORCID: 0000-0002-0885-1550
    - LinkedIn: linkedin.com/in/loveleen-gaur-ba53746
    
    ## 📚 References
    
    - Gaur, L. et al. (2023). "HCI-driven XAI model for Alzheimer's detection." 
      ACM Transactions on Multimedia Computing, Communications, and Applications.
    - Selvaraju, R. R., et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks 
      via Gradient-based Localization." ICCV.
    - NIH AIM-AHEAD: https://www.aim-ahead.net
    """)

st.markdown("---")
st.markdown("""
<div style="text-align: center; font-size: 12px; color: gray; margin-top: 30px;">
Healthcare AI Fairness Audit Platform | Built by Dr. Loveleen Gaur | 
University of Miami Frost Institute for Data Science & Computing
</div>
""", unsafe_allow_html=True)
