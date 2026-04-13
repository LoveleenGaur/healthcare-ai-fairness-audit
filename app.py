import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, confusion_matrix
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
    .info-box {
        background-color: #e3f2fd;
        border-left: 4px solid #1976d2;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3e0;
        border-left: 4px solid #f57c00;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #e8f5e9;
        border-left: 4px solid #388e3c;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .error-box {
        background-color: #ffebee;
        border-left: 4px solid #d32f2f;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.markdown("# ⚙️ Navigation")
st.sidebar.markdown("---")

tab = st.sidebar.radio(
    "What would you like to do?",
    ["📊 See Demo", "📤 Test With Your Data", "❓ How It Works", "👤 About"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Created by:**
Dr. Loveleen Gaur

📧 gaurloveleen@yahoo.com
🏆 ORCID: 0000-0002-0885-1550
""")

# ============================================================================
# HEADER
# ============================================================================

st.markdown("""
<div class="header">
    <h1>⚕️ Is Your Medical AI Fair?</h1>
    <p><strong>Check if your AI treats all patients equally</strong></p>
    <p style="font-size: 12px; margin-top: 10px;">
    This tool finds hidden bias in medical AI before it harms patients
    </p>
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
        
        try:
            tn, fp, fn, tp = confusion_matrix(group_true, group_pred, labels=[0, 1]).ravel()
        except:
            tn, fp, fn, tp = 0, 0, 0, len(group_true)
        
        positive_rate = group_pred.mean()
        accuracy = accuracy_score(group_true, group_pred)
        fpr = fp / (tn + fp) if (tn + fp) > 0 else np.nan
        fnr = fn / (tp + fn) if (tp + fn) > 0 else np.nan
        
        results.append({
            'group': group,
            'n_samples': len(group_true),
            'accuracy': accuracy,
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
        tests['Equal Predictions'] = {
            'disparity': disparity,
            'pass': disparity < threshold,
            'description': f'{disparity:.1%} difference'
        }
    
    if 'fpr' in stratified_df.columns and len(stratified_df) > 0:
        fprs = stratified_df['fpr'].dropna().values
        if len(fprs) > 0:
            fpr_disparity = fprs.max() - fprs.min()
            tests['Equal Errors'] = {
                'disparity': fpr_disparity,
                'pass': fpr_disparity < threshold,
                'description': f'{fpr_disparity:.1%} difference'
            }
    
    if 'accuracy' in stratified_df.columns and len(stratified_df) > 0:
        acc_disparity = stratified_df['accuracy'].max() - stratified_df['accuracy'].min()
        tests['Equal Accuracy'] = {
            'disparity': acc_disparity,
            'pass': acc_disparity < threshold,
            'description': f'{acc_disparity:.1%} difference'
        }
    
    return tests

def generate_recommendation(tests):
    """Generate overall deployment recommendation."""
    if not tests:
        return "⚠️ Not enough data"
    
    all_pass = all(test.get('pass', False) for test in tests.values())
    high_count = sum(1 for test in tests.values() if not test.get('pass', False))
    
    if all_pass:
        return "✅ SAFE TO USE - AI treats all patients fairly"
    elif high_count <= 1:
        return "⚠️ NEEDS FIXING - Some fairness issues found"
    else:
        return "🛑 DO NOT USE - Serious bias detected"

# ============================================================================
# DEMO MODE
# ============================================================================

if tab == "📊 See Demo":
    st.markdown("## 🎯 Real Example: Medical AI With Hidden Bias")
    
    st.markdown("""
    <div class="info-box">
    <b>What you're about to see:</b> A medical AI that seems 70% accurate overall, 
    but works much better for young patients (95%) than elderly patients (40%). 
    This hidden bias would harm elderly patients!
    </div>
    """, unsafe_allow_html=True)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 200
    
    predictions = []
    age_groups = []
    
    for i in range(n_samples):
        age = np.random.choice(['Young (45-55)', 'Middle (56-65)', 'Elderly (66+)'])
        
        if age == 'Elderly (66+)':
            pred = np.random.choice([0, 1], p=[0.65, 0.35])
        elif age == 'Middle (56-65)':
            pred = np.random.choice([0, 1], p=[0.55, 0.45])
        else:
            pred = np.random.choice([0, 1], p=[0.35, 0.65])
        
        predictions.append(pred)
        age_groups.append(age)
    
    predictions = np.array(predictions)
    ground_truth = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
    
    demographics = pd.DataFrame({'age_group': age_groups})
    
    # Overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Patients", len(predictions))
    with col2:
        st.metric("With Disease", predictions.sum())
    with col3:
        st.metric("Overall Accuracy", f"{accuracy_score(ground_truth, predictions):.1%}")
    
    st.markdown("---")
    
    # Explain what we're checking
    st.markdown("## ❓ What Are We Checking?")
    st.markdown("""
    We're checking if the AI works equally well for all age groups. 
    If it works great for young people but badly for elderly people, that's BIAS.
    """)
    
    st.markdown("---")
    
    # Step 1
    st.markdown("## Step 1️⃣: Select What To Check")
    
    st.markdown("""
    <div class="info-box">
    <b>In real life, you would choose:</b> Do you want to check if AI is fair to different ages? 
    Different genders? Different races? Let's check age groups.
    </div>
    """, unsafe_allow_html=True)
    
    protected_attr = st.selectbox(
        "What should we check for fairness?",
        ['age_group'],
        help="In real use, you'd pick: age, gender, race, etc."
    )
    
    # Explain threshold
    st.markdown("#### How much difference is OK?")
    st.markdown("If the AI is 95% accurate for young people but 85% accurate for elderly, that's a 10% difference. Is that fair?")
    
    threshold = st.slider(
        "Maximum acceptable difference:",
        min_value=0.05,
        max_value=0.25,
        value=0.10,
        step=0.01,
        help="Default is 10% - differences larger than this are unfair"
    )
    
    st.markdown("---")
    
    # Step 2
    st.markdown("## Step 2️⃣: Analyze Fairness")
    st.markdown("Let me check how the AI performs for each age group...")
    
    stratified = compute_fairness_metrics(predictions, ground_truth, demographics, protected_attr)
    tests = fairness_tests(stratified, threshold=threshold)
    recommendation = generate_recommendation(tests)
    
    # Show results in simple way
    st.markdown("### 📊 Results by Age Group")
    
    col1, col2, col3 = st.columns(3)
    for idx, row in stratified.iterrows():
        if idx == 0:
            with col1:
                st.markdown(f"""
                <div class="{'success-box' if row['accuracy'] > 0.80 else 'warning-box'}">
                <b>{row['group']}</b><br>
                Works well: {row['accuracy']:.0%}
                </div>
                """, unsafe_allow_html=True)
        elif idx == 1:
            with col2:
                st.markdown(f"""
                <div class="{'success-box' if row['accuracy'] > 0.80 else 'warning-box'}">
                <b>{row['group']}</b><br>
                Works well: {row['accuracy']:.0%}
                </div>
                """, unsafe_allow_html=True)
        else:
            with col3:
                st.markdown(f"""
                <div class="{'success-box' if row['accuracy'] > 0.80 else 'error-box'}">
                <b>{row['group']}</b><br>
                Works well: {row['accuracy']:.0%}
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Step 3 - Visualizations
    st.markdown("## Step 3️⃣: See The Bias Visually")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Accuracy by Age")
        st.markdown("How well does the AI work for each age group?")
        
        fig1 = go.Figure()
        colors = ['#27ae60' if x > 0.80 else '#f39c12' if x > 0.60 else '#e74c3c' 
                  for x in stratified['accuracy']]
        
        fig1.add_trace(go.Bar(
            x=stratified['group'],
            y=stratified['accuracy'],
            marker=dict(color=colors),
            text=[f"{x:.0%}" for x in stratified['accuracy']],
            textposition='outside'
        ))
        fig1.add_hline(y=threshold, line_dash="dash", line_color="red", 
                      annotation_text="Fair threshold")
        fig1.update_layout(
            title="Does AI work equally for all?",
            xaxis_title="Age Group",
            yaxis_title="Accuracy (how often AI is right)",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.markdown("### Who Gets Diagnosed")
        st.markdown("What % of each age group does the AI say 'has disease'?")
        
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=stratified['group'],
            y=stratified['positive_rate'],
            marker=dict(color='#3498db'),
            text=[f"{x:.0%}" for x in stratified['positive_rate']],
            textposition='outside'
        ))
        fig2.update_layout(
            title="Does AI diagnose everyone equally?",
            xaxis_title="Age Group",
            yaxis_title="% Diagnosed as having disease",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("---")
    
    # Step 4 - Explain findings
    st.markdown("## Step 4️⃣: What Does This Mean?")
    
    for test_name, test_result in tests.items():
        if test_name == 'Equal Accuracy':
            st.markdown("#### Is accuracy the same for all ages?")
            st.markdown(f"""
            <div class="{'success-box' if test_result['pass'] else 'error-box'}">
            <b>Finding: {test_result['description']}</b><br><br>
            <b>In Plain English:</b> 
            """, unsafe_allow_html=True)
            
            if test_result['pass']:
                st.markdown("✅ Good! The AI works about the same for all age groups. Fair!")
            else:
                st.markdown("❌ Problem! The AI works much better for some ages than others. This is BIAS!")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        elif test_name == 'Equal Predictions':
            st.markdown("#### Does the AI diagnose everyone equally?")
            st.markdown(f"""
            <div class="{'success-box' if test_result['pass'] else 'error-box'}">
            <b>Finding: {test_result['description']}</b><br><br>
            <b>In Plain English:</b> 
            """, unsafe_allow_html=True)
            
            if test_result['pass']:
                st.markdown("✅ Good! All age groups get diagnosed at about the same rate.")
            else:
                st.markdown("❌ Problem! Some age groups get diagnosed much more than others. This is BIAS!")
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Final recommendation
    st.markdown("## Step 5️⃣: Should We Use This AI?")
    
    if "SAFE" in recommendation:
        st.markdown(f"""
        <div class="success-box">
        <h3>✅ {recommendation}</h3>
        The AI treats young and elderly patients fairly. It's ready to use!
        </div>
        """, unsafe_allow_html=True)
    elif "NEEDS" in recommendation:
        st.markdown(f"""
        <div class="warning-box">
        <h3>⚠️ {recommendation}</h3>
        The AI has some fairness issues. We need to improve it before using it.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="error-box">
        <h3>🛑 {recommendation}</h3>
        The AI has serious bias. We cannot use it - it would harm elderly patients!
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# UPLOAD MODE
# ============================================================================

elif tab == "📤 Test With Your Data":
    st.markdown("## 🔍 Test Your Own Medical AI")
    
    st.markdown("""
    <div class="info-box">
    <b>Do you have your own AI model?</b> Upload the results here to check if it's fair!
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### What You Need:")
    st.markdown("""
    A spreadsheet (CSV) with these columns:
    - **prediction** (what your AI said: 0 or 1)
    - **ground_truth** (what actually happened: 0 or 1)
    - **age_group** (or any demographic like: gender, race, etc.)
    """)
    
    st.markdown("### Example:")
    st.markdown("""
    | prediction | ground_truth | age_group |
    |------------|--------------|-----------|
    | 1          | 1            | Young     |
    | 0          | 0            | Young     |
    | 1          | 0            | Elderly   |
    | 0          | 1            | Elderly   |
    """)
    
    uploaded_file = st.file_uploader(
        "📁 Upload your CSV file",
        type=['csv'],
        help="Click to choose a file from your computer"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.success("✅ File loaded! Now let's configure it...")
            
            st.markdown("---")
            st.markdown("### Step 1: Tell Me Your Column Names")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                pred_col = st.selectbox(
                    "Which column is the AI prediction?",
                    df.columns,
                    help="Should contain 0 or 1 values"
                )
            with col2:
                truth_col = st.selectbox(
                    "Which column is the truth?",
                    df.columns,
                    help="What actually happened (0 or 1)"
                )
            with col3:
                demo_cols = st.multiselect(
                    "Which column shows groups?",
                    df.columns,
                    help="Like: age_group, gender, etc."
                )
            
            st.markdown("---")
            st.markdown("### Step 2: How Fair Should It Be?")
            
            threshold = st.slider(
                "Maximum difference allowed:",
                min_value=0.05,
                max_value=0.25,
                value=0.10,
                step=0.01,
                help="10% is the standard fairness threshold"
            )
            
            if st.button("🔍 Check Fairness", type="primary"):
                
                predictions = df[pred_col].values
                ground_truth = df[truth_col].values
                
                st.markdown("---")
                st.markdown("### 📊 Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Patients Analyzed", len(predictions))
                with col2:
                    st.metric("Overall Accuracy", f"{accuracy_score(ground_truth, predictions):.1%}")
                with col3:
                    st.metric("Diagnosed as Sick", predictions.sum())
                
                st.markdown("---")
                
                for demo_col in demo_cols:
                    st.markdown(f"### Results by {demo_col}")
                    
                    stratified = compute_fairness_metrics(predictions, ground_truth, df, demo_col)
                    tests = fairness_tests(stratified, threshold)
                    
                    st.dataframe(
                        stratified[['group', 'n_samples', 'accuracy', 'positive_rate']].round(3),
                        use_container_width=True
                    )
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig1 = go.Figure()
                        colors = ['#27ae60' if x > 0.80 else '#f39c12' if x > 0.60 else '#e74c3c' 
                                  for x in stratified['accuracy']]
                        fig1.add_trace(go.Bar(x=stratified['group'], y=stratified['accuracy'],
                                             marker=dict(color=colors), text=[f"{x:.0%}" for x in stratified['accuracy']],
                                             textposition='outside'))
                        fig1.update_layout(title=f"Accuracy by {demo_col}", height=400)
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    with col2:
                        fig2 = go.Figure()
                        fig2.add_trace(go.Bar(x=stratified['group'], y=stratified['positive_rate'],
                                             marker=dict(color='#3498db'), text=[f"{x:.0%}" for x in stratified['positive_rate']],
                                             textposition='outside'))
                        fig2.update_layout(title=f"Diagnosis Rate by {demo_col}", height=400)
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    st.markdown("#### Fairness Check:")
                    for test_name, test_result in tests.items():
                        if test_result['pass']:
                            st.markdown(f"✅ {test_name}: Fair ({test_result['description']})")
                        else:
                            st.markdown(f"❌ {test_name}: Unfair ({test_result['description']})")
                    
                    st.markdown("---")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("Make sure your CSV has columns named: prediction, ground_truth, and a group column")

# ============================================================================
# HOW IT WORKS
# ============================================================================

elif tab == "❓ How It Works":
    st.markdown("## 🎓 Understanding AI Fairness (In Simple Words)")
    
    st.markdown("""
    <div class="info-box">
    <b>What is AI bias?</b> When an AI works better for some people than others.
    <br><br>
    <b>Example:</b> An AI diagnoses heart disease in men 95% of the time, 
    but only 70% of the time in women. This is BIAS and UNFAIR!
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### Why Does This Matter?")
    st.markdown("""
    When medical AI is biased:
    - ❌ Women get fewer diagnoses (they suffer)
    - ❌ Hospital gets sued
    - ❌ Doctors trust the AI and miss cases
    - ❌ Violates medical regulations
    """)
    
    st.markdown("---")
    
    st.markdown("### How Does This App Help?")
    st.markdown("""
    This app checks **3 things**:
    """)
    
    st.markdown("#### 1️⃣ Equal Accuracy")
    st.markdown("""
    <div class="info-box">
    <b>Question:</b> Does the AI work equally well for all age groups?<br><br>
    <b>Example:</b><br>
    ✅ FAIR: Works 90% for young AND 88% for elderly (2% difference)<br>
    ❌ UNFAIR: Works 95% for young BUT 70% for elderly (25% difference)
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### 2️⃣ Equal Diagnosis Rate")
    st.markdown("""
    <div class="info-box">
    <b>Question:</b> Does the AI diagnose equally across groups?<br><br>
    <b>Example:</b><br>
    ✅ FAIR: Diagnoses 60% of young AND 58% of elderly<br>
    ❌ UNFAIR: Diagnoses 80% of young BUT 30% of elderly (many elderly missed!)
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### 3️⃣ Equal Error Rate")
    st.markdown("""
    <div class="info-box">
    <b>Question:</b> Does the AI make similar mistakes for all groups?<br><br>
    <b>Example:</b><br>
    ✅ FAIR: False alarms 5% for young AND 6% for elderly<br>
    ❌ UNFAIR: False alarms 5% for young BUT 20% for elderly (elderly get more scares)
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### The Fairness Threshold (Why 10%?)")
    st.markdown("""
    We say differences less than 10% are "acceptable" because:
    - ✅ Small natural variation exists
    - ✅ 10% is the industry standard
    - ✅ More than 10% means real bias
    - ✅ You can change this slider!
    """)
    
    st.markdown("---")
    
    st.markdown("### Real Hospital Example")
    st.markdown("""
    **Scenario:** A hospital wants to use an AI to predict stroke risk.
    
    **What the AI does:**
    - For 60-year-olds: 95% accurate ✅
    - For 80-year-olds: 60% accurate ❌
    
    **Why this is a problem:**
    - Elderly people are MORE likely to have strokes
    - But the AI is WORSE at detecting them!
    - Elderly patients get missed diagnoses
    - People have strokes and don't get help
    
    **What this app would say:** 🛑 DO NOT USE - Serious bias detected
    
    **What the hospital should do:** Retrain the AI with more elderly patients until it's fair.
    """)

# ============================================================================
# ABOUT MODE
# ============================================================================

elif tab == "👤 About":
    st.markdown("## 👋 About This Tool")
    
    st.markdown("""
    This tool was built to solve a real problem: **Medical AI that looks fair but isn't.**
    
    ### 🎯 Our Goal
    Protect patients by finding AI bias BEFORE it harms them.
    
    ### 👨‍🔬 Who Built This?
    **Dr. Loveleen Gaur**
    - AI researcher and ethicist
    - 130+ published research papers
    - Focus: Making AI fair and trustworthy
    - Email: gaurloveleen@yahoo.com
    
    ### 🏆 Credentials
    - Top 2% of scientists globally
    - 30+ books on AI ethics
    - Published in major journals
    
    ### 📚 Based On Research
    This tool uses fairness standards from:
    - Academic peer-reviewed papers
    - Hospital regulations
    - Ethical AI guidelines
    
    ### 💡 How It Works
    1. You upload your AI predictions
    2. We split data by groups (age, gender, etc.)
    3. We check if accuracy is equal
    4. We tell you if bias exists
    5. We recommend: Safe to use? Or fix it first?
    
    ### 🔒 Privacy
    - Your data is analyzed here
    - Nothing is saved
    - Nothing is shared
    - Your data stays private
    """)

st.markdown("---")
st.markdown("""
<div style="text-align: center; font-size: 11px; color: gray; margin-top: 30px;">
Made to protect patients from biased medical AI ❤️
</div>
""", unsafe_allow_html=True)
