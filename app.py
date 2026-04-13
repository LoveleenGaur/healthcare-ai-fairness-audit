import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="AI Fairness Audit Platform",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling - Enhanced
st.markdown("""
    <style>
    .main { padding: 20px; background-color: #f8f9fa; }
    .header {
        background: linear-gradient(135deg, #1f77b4 0%, #2ca02c 100%);
        padding: 25px;
        border-radius: 10px;
        color: white;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .pass { color: #27ae60; font-weight: bold; }
    .warning { color: #f39c12; font-weight: bold; }
    .fail { color: #e74c3c; font-weight: bold; }
    .section-header {
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.markdown("# ⚙️ Configuration")
st.sidebar.markdown("---")

tab = st.sidebar.radio(
    "Select Mode:",
    ["📊 Demo", "📤 Upload Data", "📈 Advanced Analysis", "ℹ️ About"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Dr. Loveleen Gaur**  
AI Ethics & Governance  
📧 gaurloveleen@yahoo.com  
ORCID: 0000-0002-0885-1550
""")

# ============================================================================
# HEADER
# ============================================================================

st.markdown("""
<div class="header">
    <h1>⚕️ AI Fairness Audit Platform</h1>
    <p><strong>Detecting Demographic Bias in Medical AI Models</strong></p>
    <p style="font-size: 12px; margin-top: 10px;">
    Research-backed fairness assessment tool for healthcare AI systems
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# ENHANCED HELPER FUNCTIONS
# ============================================================================

@st.cache_data
def compute_fairness_metrics(predictions, ground_truth, demographics, protected_attr):
    """Compute comprehensive fairness metrics stratified by demographic group."""
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
        
        # Metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        positive_rate = group_pred.mean()
        accuracy = accuracy_score(group_true, group_pred)
        precision = precision_score(group_true, group_pred, zero_division=0)
        recall = recall_score(group_true, group_pred, zero_division=0)
        f1 = f1_score(group_true, group_pred, zero_division=0)
        fpr = fp / (tn + fp) if (tn + fp) > 0 else np.nan
        fnr = fn / (tp + fn) if (tp + fn) > 0 else np.nan
        
        # Disparate Impact Ratio
        if positive_rate > 0:
            di_ratio = positive_rate / (group_pred[~mask].mean() + 1e-6) if len(group_pred[~mask]) > 0 else 1.0
        else:
            di_ratio = 1.0
        
        results.append({
            'group': group,
            'n_samples': len(group_true),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'positive_rate': positive_rate,
            'fpr': fpr,
            'fnr': fnr,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        })
    
    return pd.DataFrame(results)

def fairness_tests(stratified_df, threshold=0.10):
    """Run comprehensive fairness tests with detailed reporting."""
    tests = {}
    
    # Test 1: Demographic Parity
    if 'positive_rate' in stratified_df.columns and len(stratified_df) > 0:
        pos_rates = stratified_df['positive_rate'].values
        disparity = pos_rates.max() - pos_rates.min()
        tests['Demographic Parity'] = {
            'disparity': disparity,
            'pass': disparity < threshold,
            'severity': 'CRITICAL' if disparity > 0.20 else 'HIGH' if disparity > threshold else 'LOW',
            'description': f'Positive rate disparity: {disparity:.1%}',
            'acceptable': disparity < threshold
        }
    
    # Test 2: Equalized Odds (FPR)
    if 'fpr' in stratified_df.columns and len(stratified_df) > 0:
        fprs = stratified_df['fpr'].dropna().values
        if len(fprs) > 0:
            fpr_disparity = fprs.max() - fprs.min()
            tests['Equalized Odds (FPR)'] = {
                'disparity': fpr_disparity,
                'pass': fpr_disparity < threshold,
                'severity': 'CRITICAL' if fpr_disparity > 0.20 else 'HIGH' if fpr_disparity > threshold else 'LOW',
                'description': f'False Positive Rate disparity: {fpr_disparity:.1%}',
                'acceptable': fpr_disparity < threshold
            }
    
    # Test 3: Accuracy Parity
    if 'accuracy' in stratified_df.columns and len(stratified_df) > 0:
        acc_disparity = stratified_df['accuracy'].max() - stratified_df['accuracy'].min()
        tests['Accuracy Parity'] = {
            'disparity': acc_disparity,
            'pass': acc_disparity < threshold,
            'severity': 'CRITICAL' if acc_disparity > 0.20 else 'HIGH' if acc_disparity > threshold else 'LOW',
            'description': f'Accuracy disparity: {acc_disparity:.1%}',
            'acceptable': acc_disparity < threshold
        }
    
    # Test 4: Equal Opportunity (Recall/TPR)
    if 'recall' in stratified_df.columns and len(stratified_df) > 0:
        recalls = stratified_df['recall'].values
        recall_disparity = recalls.max() - recalls.min()
        tests['Equal Opportunity'] = {
            'disparity': recall_disparity,
            'pass': recall_disparity < threshold,
            'severity': 'CRITICAL' if recall_disparity > 0.20 else 'HIGH' if recall_disparity > threshold else 'LOW',
            'description': f'True Positive Rate disparity: {recall_disparity:.1%}',
            'acceptable': recall_disparity < threshold
        }
    
    return tests

def generate_recommendation(tests):
    """Generate deployment recommendation based on fairness tests."""
    if not tests:
        return "⚠️ INSUFFICIENT DATA"
    
    all_pass = all(test.get('acceptable', False) for test in tests.values())
    critical_count = sum(1 for test in tests.values() if test.get('severity') == 'CRITICAL')
    high_count = sum(1 for test in tests.values() if test.get('severity') == 'HIGH')
    
    if all_pass:
        return "✅ APPROVED FOR DEPLOYMENT (with ongoing monitoring)"
    elif critical_count == 0 and high_count <= 2:
        return "⚠️ CONDITIONAL APPROVAL (address fairness gaps before deployment)"
    else:
        return "🛑 DO NOT DEPLOY (significant fairness concerns)"

# ============================================================================
# DEMO MODE
# ============================================================================

if tab == "📊 Demo":
    st.markdown("### Demo: Model Fairness Audit")
    st.markdown("*Interactive demonstration with synthetic healthcare AI model data*")
    
    # Synthetic data generation
    np.random.seed(42)
    n_samples = 200
    
    predictions = []
    age_groups = []
    genders = []
    
    for i in range(n_samples):
        age = np.random.choice(['55-65', '65-75', '75+'])
        gender = np.random.choice(['M', 'F'])
        
        # Simulate intentional bias for demo
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
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Samples", len(predictions))
    with col2:
        st.metric("Positive Cases", predictions.sum())
    with col3:
        st.metric("Overall Accuracy", f"{accuracy_score(ground_truth, predictions):.1%}")
    with col4:
        overall_precision = precision_score(ground_truth, predictions, zero_division=0)
        st.metric("Overall Precision", f"{overall_precision:.1%}")
    
    st.markdown("---")
    
    # Demographic selection
    st.markdown("<div class='section-header'>Step 1: Select Demographic Attribute</div>", unsafe_allow_html=True)
    protected_attr = st.selectbox(
        "Which demographic group to audit for bias?",
        ['age_group', 'gender'],
        help="Select demographic attribute to analyze"
    )
    
    # Fairness threshold
    col1, col2 = st.columns(2)
    with col1:
        threshold = st.slider(
            "Fairness Threshold (acceptable disparity)",
            min_value=0.05,
            max_value=0.25,
            value=0.10,
            step=0.01,
            help="Maximum acceptable difference between groups"
        )
    
    st.markdown("---")
    
    # Compute metrics
    st.markdown("<div class='section-header'>Step 2: Fairness Analysis</div>", unsafe_allow_html=True)
    
    stratified = compute_fairness_metrics(predictions, ground_truth, demographics, protected_attr)
    tests = fairness_tests(stratified, threshold=threshold)
    recommendation = generate_recommendation(tests)
    
    # Display detailed metrics table
    st.markdown(f"**Performance Breakdown by {protected_attr}:**")
    display_cols = ['group', 'n_samples', 'accuracy', 'precision', 'recall', 'positive_rate']
    st.dataframe(
        stratified[display_cols].round(3),
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("---")
    
    # Visualizations
    st.markdown("<div class='section-header'>Step 3: Bias Visualization</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Accuracy by group
        fig1 = go.Figure()
        colors = ['#27ae60' if x > 0.75 else '#f39c12' if x > 0.65 else '#e74c3c' 
                  for x in stratified['accuracy']]
        
        fig1.add_trace(go.Bar(
            x=stratified['group'],
            y=stratified['accuracy'],
            marker=dict(color=colors),
            text=[f"{x:.1%}" for x in stratified['accuracy']],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Accuracy: %{y:.1%}<extra></extra>'
        ))
        fig1.add_hline(y=threshold, line_dash="dash", line_color="red",
                      annotation_text="Threshold")
        fig1.update_layout(
            title=f"Accuracy by {protected_attr}",
            xaxis_title=protected_attr,
            yaxis_title="Accuracy",
            height=400,
            showlegend=False,
            hovermode='x unified'
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Positive prediction rate
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=stratified['group'],
            y=stratified['positive_rate'],
            marker=dict(color='#3498db'),
            text=[f"{x:.1%}" for x in stratified['positive_rate']],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Positive Rate: %{y:.1%}<extra></extra>'
        ))
        fig2.update_layout(
            title="Positive Prediction Rate (Demographic Parity Check)",
            xaxis_title=protected_attr,
            yaxis_title="Positive Rate",
            height=400,
            showlegend=False,
            hovermode='x unified'
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("---")
    
    # Fairness test results
    st.markdown("<div class='section-header'>Step 4: Fairness Test Results</div>", unsafe_allow_html=True)
    
    for test_name, test_result in tests.items():
        col1, col2, col3 = st.columns([1, 2, 2])
        
        with col1:
            if test_result['acceptable']:
                st.markdown("✅ **PASS**")
            else:
                st.markdown(f"⚠️ **{test_result['severity']}**")
        
        with col2:
            st.markdown(f"**{test_name}**")
        
        with col3:
            st.markdown(test_result['description'])
    
    st.markdown("---")
    
    # Deployment recommendation
    st.markdown("<div class='section-header'>Step 5: Deployment Recommendation</div>", unsafe_allow_html=True)
    
    if "APPROVED" in recommendation:
        color, emoji = "green", "✅"
    elif "CONDITIONAL" in recommendation:
        color, emoji = "orange", "⚠️"
    else:
        color, emoji = "red", "🛑"
    
    st.markdown(f"""
    <div style="background-color: {color}20; border-left: 4px solid {color}; padding: 15px; border-radius: 5px;">
    <h3 style="color: {color}; margin-top: 0;">{emoji} {recommendation}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Interpretation guide
    with st.expander("📖 How to Interpret Results"):
        st.info("""
        **Key Fairness Metrics:**
        
        🔸 **Demographic Parity**: Does the model make positive predictions at equal rates across groups?
        
        🔸 **Equalized Odds**: Are error rates similar across groups? (False Positives & False Negatives)
        
        🔸 **Accuracy Parity**: Is model accuracy consistent across demographic groups?
        
        🔸 **Equal Opportunity**: Do all groups have similar True Positive Rates?
        
        **Red Flags:**
        - Different accuracy across groups → Model unreliable for certain populations
        - Different prediction rates → Systematic over/under-prediction for some groups
        - Different error rates → Higher cost of mistakes for some groups
        """)

# ============================================================================
# UPLOAD MODE
# ============================================================================

elif tab == "📤 Upload Data":
    st.markdown("### Audit Your Model")
    st.markdown("Upload predictions and demographic data for custom fairness analysis")
    
    st.info("""
    **Required Format:**
    - CSV file with columns for: predictions, ground truth, demographics
    - Predictions: 0 or 1 (binary classification)
    - Ground truth: actual outcomes (0 or 1)
    - Demographics: age, gender, ethnicity, etc.
    """)
    
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.success("✅ File loaded!")
            st.dataframe(df.head(10), use_container_width=True)
            
            st.markdown("---")
            st.markdown("### Configure Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                pred_col = st.selectbox("Prediction column:", df.columns)
            with col2:
                truth_col = st.selectbox("Ground truth column:", df.columns)
            with col3:
                demo_cols = st.multiselect("Demographics:", df.columns)
            
            threshold = st.slider("Fairness threshold:", 0.05, 0.25, 0.10)
            
            if st.button("🔍 Run Analysis", type="primary"):
                
                predictions = df[pred_col].values
                ground_truth = df[truth_col].values
                
                st.markdown("---")
                st.markdown("### Audit Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Samples", len(predictions))
                with col2:
                    st.metric("Accuracy", f"{accuracy_score(ground_truth, predictions):.1%}")
                with col3:
                    st.metric("Positive Cases", predictions.sum())
                
                st.markdown("---")
                
                # Analyze each demographic
                for demo_col in demo_cols:
                    st.markdown(f"### Analysis by {demo_col}")
                    
                    stratified = compute_fairness_metrics(predictions, ground_truth, df, demo_col)
                    tests = fairness_tests(stratified, threshold)
                    
                    st.dataframe(
                        stratified[['group', 'n_samples', 'accuracy', 'recall']].round(3),
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
                        fig2.add_trace(go.Bar(x=stratified['group'], y=stratified['recall'],
                                             marker=dict(color='#3498db')))
                        fig2.update_layout(title=f"Recall by {demo_col}", height=400)
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    st.markdown("**Fairness Tests:**")
                    for test_name, test_result in tests.items():
                        status = "✅ PASS" if test_result['acceptable'] else f"⚠️ {test_result['severity']}"
                        st.markdown(f"{status} - {test_name}: {test_result['description']}")
                    
                    st.markdown("---")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")

# ============================================================================
# ADVANCED ANALYSIS
# ============================================================================

elif tab == "📈 Advanced Analysis":
    st.markdown("### Advanced Fairness Analysis")
    st.markdown("Deeper statistical analysis and sensitivity testing")
    
    st.info("""
    **Available Tools:**
    - Detailed confusion matrices by demographic group
    - ROC curves and AUC analysis
    - Threshold sensitivity analysis
    - Intersectional fairness (multiple demographics)
    """)
    
    st.markdown("""
    **How to use:**
    1. Upload your data in the "Upload Data" tab
    2. Return here for advanced metrics
    3. Adjust thresholds to see impact on fairness
    
    Advanced features coming soon...
    """)

# ============================================================================
# ABOUT MODE
# ============================================================================

elif tab == "ℹ️ About":
    st.markdown("""
    # About This Platform
    
    ## 🎯 Purpose
    
    Comprehensive fairness auditing tool for AI/ML models in healthcare and high-stakes applications.
    
    ## 🔬 Research Foundation
    
    - **Author:** Dr. Loveleen Gaur
    - **Expertise:** AI Ethics, Explainable AI, Healthcare AI
    - **Credentials:** 
      - USCIS EB1A Extraordinary Ability in AI
      - Top 2% Scientists (2024-2025)
      - 130+ peer-reviewed publications
      - 30+ published books on AI ethics
    
    ## 📊 Fairness Metrics Explained
    
    ### Demographic Parity
    Do groups get predicted positive at equal rates?
    - ✓ Threshold: < 10% difference
    - Ensures equal treatment in predictions
    
    ### Equalized Odds
    Are error rates similar across groups?
    - ✓ Threshold: < 10% difference
    - Ensures equal chance of correct prediction
    
    ### Accuracy Parity
    Is model accuracy consistent?
    - ✓ Threshold: < 10% difference
    - Ensures reliable performance for all groups
    
    ### Equal Opportunity
    Do groups have similar True Positive Rates?
    - ✓ Threshold: < 10% difference
    - Ensures equal access to positive outcomes
    
    ## 🏥 Use Cases
    
    1. **Pre-deployment Audit** — Screen models before use
    2. **Regulatory Compliance** — Meet institutional fairness requirements
    3. **Model Selection** — Compare fairness of different models
    4. **Training Data Analysis** — Identify underrepresented populations
    5. **Continuous Monitoring** — Track fairness over time
    
    ## 💡 Key Insights
    
    ✓ Bias is often hidden in accuracy metrics  
    ✓ High overall accuracy ≠ Fair performance  
    ✓ Different groups may have different error costs  
    ✓ Regular auditing catches emerging bias early  
    ✓ Fairness requires deliberate design
    
    ## 📧 Contact
    
    **Dr. Loveleen Gaur**
    - Email: gaurloveleen@yahoo.com
    - ORCID: 0000-0002-0885-1550
    
    ## 📚 References
    
    - Gaur, L. et al. (2023). "HCI-driven XAI model for disease detection." 
      ACM Transactions on Multimedia Computing, Communications, and Applications.
    - Selvaraju, R. R., et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks." ICCV.
    - Fairness definitions: Barocas, S., Hardt, M., & Narayanan, A. (2019).
    """)

st.markdown("---")
st.markdown("""
<div style="text-align: center; font-size: 11px; color: gray;">
AI Fairness Audit Platform | Built by Dr. Loveleen Gaur
</div>
""", unsafe_allow_html=True)
