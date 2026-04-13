import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
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

# Custom styling
st.markdown("""
    <style>
    .main { padding: 20px; background-color: #f8f9fa; }
    .header {
        background: linear-gradient(135deg, #1f77b4 0%, #2ca02c 100%);
        padding: 25px;
        border-radius: 10px;
        color: white;
    }
    .pass { color: #27ae60; font-weight: bold; }
    .warning { color: #f39c12; font-weight: bold; }
    .fail { color: #e74c3c; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.markdown("# ⚙️ Configuration")
st.sidebar.markdown("---")

tab = st.sidebar.radio(
    "Select Mode:",
    ["📊 Demo", "📤 Upload Data", "ℹ️ About"]
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
</div>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

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
        
        # Confusion matrix
        try:
            tn, fp, fn, tp = confusion_matrix(group_true, group_pred, labels=[0, 1]).ravel()
        except:
            tn, fp, fn, tp = 0, 0, 0, len(group_true)
        
        # Metrics
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
    
    if 'positive_rate' in stratified_df.columns and len(stratified_df) > 0:
        pos_rates = stratified_df['positive_rate'].values
        disparity = pos_rates.max() - pos_rates.min()
        tests['Demographic Parity'] = {
            'disparity': disparity,
            'pass': disparity < threshold,
            'description': f'Positive rate disparity: {disparity:.1%}'
        }
    
    if 'fpr' in stratified_df.columns and len(stratified_df) > 0:
        fprs = stratified_df['fpr'].dropna().values
        if len(fprs) > 0:
            fpr_disparity = fprs.max() - fprs.min()
            tests['Equalized Odds (FPR)'] = {
                'disparity': fpr_disparity,
                'pass': fpr_disparity < threshold,
                'description': f'False Positive Rate disparity: {fpr_disparity:.1%}'
            }
    
    if 'accuracy' in stratified_df.columns and len(stratified_df) > 0:
        acc_disparity = stratified_df['accuracy'].max() - stratified_df['accuracy'].min()
        tests['Accuracy Parity'] = {
            'disparity': acc_disparity,
            'pass': acc_disparity < threshold,
            'description': f'Accuracy disparity: {acc_disparity:.1%}'
        }
    
    return tests

def generate_recommendation(tests):
    """Generate overall deployment recommendation."""
    if not tests:
        return "⚠️ INSUFFICIENT DATA"
    
    all_pass = all(test.get('pass', False) for test in tests.values())
    high_count = sum(1 for test in tests.values() if not test.get('pass', False))
    
    if all_pass:
        return "✅ APPROVED FOR DEPLOYMENT"
    elif high_count <= 1:
        return "⚠️ CONDITIONAL APPROVAL"
    else:
        return "🛑 DO NOT DEPLOY"

# ============================================================================
# DEMO MODE
# ============================================================================

if tab == "📊 Demo":
    st.markdown("### Demo: Model Fairness Audit")
    
    # Generate synthetic data
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
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples", len(predictions))
    with col2:
        st.metric("Positive Cases", predictions.sum())
    with col3:
        st.metric("Overall Accuracy", f"{accuracy_score(ground_truth, predictions):.1%}")
    
    st.markdown("---")
    
    # Demographic selection
    st.markdown("### Step 1: Select Demographic Attribute")
    protected_attr = st.selectbox(
        "Which demographic group to audit?",
        ['age_group', 'gender']
    )
    
    # Fairness threshold
    threshold = st.slider(
        "Fairness Threshold",
        min_value=0.05,
        max_value=0.25,
        value=0.10,
        step=0.01
    )
    
    st.markdown("---")
    
    # Compute metrics
    st.markdown("### Step 2: Fairness Analysis")
    
    stratified = compute_fairness_metrics(predictions, ground_truth, demographics, protected_attr)
    tests = fairness_tests(stratified, threshold=threshold)
    recommendation = generate_recommendation(tests)
    
    # Display table
    st.markdown(f"**Performance by {protected_attr}:**")
    st.dataframe(
        stratified[['group', 'n_samples', 'accuracy', 'positive_rate']].round(3),
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("---")
    
    # Visualizations
    st.markdown("### Step 3: Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = go.Figure()
        colors = ['#27ae60' if x > 0.75 else '#f39c12' if x > 0.65 else '#e74c3c' 
                  for x in stratified['accuracy']]
        
        fig1.add_trace(go.Bar(
            x=stratified['group'],
            y=stratified['accuracy'],
            marker=dict(color=colors),
            text=[f"{x:.1%}" for x in stratified['accuracy']],
            textposition='outside'
        ))
        fig1.update_layout(
            title=f"Accuracy by {protected_attr}",
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
            text=[f"{x:.1%}" for x in stratified['positive_rate']],
            textposition='outside'
        ))
        fig2.update_layout(
            title="Positive Prediction Rate",
            xaxis_title=protected_attr,
            yaxis_title="Positive Rate",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("---")
    
    # Fairness test results
    st.markdown("### Step 4: Fairness Tests")
    
    for test_name, test_result in tests.items():
        if test_result['pass']:
            st.markdown(f"✅ **{test_name}** - PASS")
        else:
            st.markdown(f"⚠️ **{test_name}** - WARNING")
        st.markdown(f"   {test_result['description']}")
    
    st.markdown("---")
    
    # Recommendation
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

# ============================================================================
# UPLOAD MODE
# ============================================================================

elif tab == "📤 Upload Data":
    st.markdown("### Audit Your Model")
    st.markdown("Upload CSV with predictions, ground_truth, and demographics")
    
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
                st.markdown("### Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Samples", len(predictions))
                with col2:
                    st.metric("Accuracy", f"{accuracy_score(ground_truth, predictions):.1%}")
                with col3:
                    st.metric("Positive", predictions.sum())
                
                st.markdown("---")
                
                for demo_col in demo_cols:
                    st.markdown(f"### Analysis by {demo_col}")
                    
                    stratified = compute_fairness_metrics(predictions, ground_truth, df, demo_col)
                    tests = fairness_tests(stratified, threshold)
                    
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
                        status = "✅ PASS" if test_result['pass'] else "⚠️ WARNING"
                        st.markdown(f"{status} - {test_name}")
                    
                    st.markdown("---")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")

# ============================================================================
# ABOUT MODE
# ============================================================================

elif tab == "ℹ️ About":
    st.markdown("""
    # About This Platform
    
    ## 🎯 Purpose
    Audit healthcare AI models for demographic bias before deployment.
    
    ## 🔬 Research Foundation
    
    **Author:** Dr. Loveleen Gaur
    - USCIS EB1A Extraordinary Ability in AI
    - Elsevier-Stanford Top 2% Scientists (2024-2025)
    - 130+ peer-reviewed publications
    - 30+ published books on AI ethics
    
    ## 📊 Fairness Metrics
    
    ### Demographic Parity
    Equal positive prediction rates across groups
    
    ### Equalized Odds
    Equal false positive rates across groups
    
    ### Accuracy Parity
    Consistent accuracy across groups
    
    ## 📧 Contact
    
    Dr. Loveleen Gaur
    - Email: gaurloveleen@yahoo.com
    - ORCID: 0000-0002-0885-1550
    """)

st.markdown("---")
st.markdown("""
<div style="text-align: center; font-size: 11px; color: gray;">
AI Fairness Audit Platform | Built by Dr. Loveleen Gaur
</div>
""", unsafe_allow_html=True)
