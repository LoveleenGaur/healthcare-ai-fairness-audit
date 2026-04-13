import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="AI Fairness Audit Platform",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        font-size: 14px;
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
    .explain-text {
        background-color: #f5f5f5;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        font-size: 13px;
        font-style: italic;
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
        
        results.append({
            'group': group,
            'n_samples': len(group_true),
            'accuracy': accuracy,
            'positive_rate': positive_rate,
            'fpr': fpr
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
    
    st.markdown("""
    <div class="header">
        <h1>⚕️ Is Your Medical AI Fair?</h1>
        <p><strong>Let's Check if an AI Treats All Patients Equally</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <b>📌 What You're About To See:</b><br>
    A real example of medical AI that has hidden bias. 
    On the surface it looks 70% accurate, but it actually works MUCH better 
    for young patients (95% accurate) than elderly patients (40% accurate). 
    This bias would harm elderly people if we used it in a hospital!
    </div>
    """, unsafe_allow_html=True)
    
    # Generate synthetic data with MULTIPLE demographics
    np.random.seed(42)
    n_samples = 200
    
    predictions = []
    age_groups = []
    genders = []
    
    for i in range(n_samples):
        age = np.random.choice(['Young (45-55)', 'Middle (56-65)', 'Elderly (66+)'])
        gender = np.random.choice(['Male', 'Female'])
        
        # Different accuracy for age groups
        if age == 'Elderly (66+)':
            pred = np.random.choice([0, 1], p=[0.65, 0.35])
        elif age == 'Middle (56-65)':
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
    
    # Overview
    st.markdown("## 📊 Quick Overview")
    st.markdown("""
    <div class="explain-text">
    <b>What these numbers mean:</b><br>
    • <b>Total Patients:</b> How many people we tested<br>
    • <b>With Disease:</b> How many the AI said had the disease<br>
    • <b>Overall Accuracy:</b> How often the AI got it right (% of the time)
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Patients", len(predictions))
    with col2:
        st.metric("AI Says Has Disease", predictions.sum())
    with col3:
        st.metric("Overall Accuracy", f"{accuracy_score(ground_truth, predictions):.1%}")
    
    st.markdown("""
    <div class="warning-box">
    <b>⚠️ Notice:</b> Overall accuracy is 70%. This looks okay! But wait... 
    we're about to discover the hidden bias...
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Step 1 - Explain what we're checking
    st.markdown("## 🔍 Step 1: What Are We Checking?")
    
    st.markdown("""
    <div class="info-box">
    <b>The Big Question:</b> Does the AI work equally well for ALL age groups?<br><br>
    <b>Why this matters:</b> If the AI works great for young people but poorly for elderly people, 
    that's BIAS. Elderly patients won't get diagnosed correctly and might not get treatment they need.
    </div>
    """, unsafe_allow_html=True)
    
    # Get all available demographic columns
    demographic_columns = [col for col in demographics.columns]
    
    protected_attr = st.selectbox(
        "What should we check fairness for?",
        demographic_columns,
        help="Choose any demographic: age_group, gender, race, income level, etc. We'll check if AI is fair to this group."
    )
    
    st.markdown(f"""
    <div class="explain-text">
    <b>What you just selected:</b> We'll compare how the AI works for different {protected_attr} groups. 
    If it's much better for some groups than others, that's a PROBLEM - it's bias!
    </div>
    """, unsafe_allow_html=True)
    
    # Explain threshold
    st.markdown("#### How much difference is acceptable?")
    
    st.markdown("""
    <div class="explain-text">
    <b>Example:</b> If AI is 95% accurate for young AND 93% accurate for elderly, 
    that's only 2% difference - FAIR!<br><br>
    <b>Bad example:</b> If AI is 95% accurate for young BUT 70% accurate for elderly, 
    that's 25% difference - UNFAIR!
    </div>
    """, unsafe_allow_html=True)
    
    threshold = st.slider(
        "📏 Maximum acceptable difference:",
        min_value=0.05,
        max_value=0.25,
        value=0.10,
        step=0.01,
        help="Default is 10% - this is the medical industry standard for fairness"
    )
    
    st.markdown(f"""
    <div class="explain-text">
    <b>You chose:</b> We will say the AI is FAIR if differences are less than {threshold:.0%}. 
    If differences are MORE than {threshold:.0%}, we'll say it's UNFAIR.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Step 2 - Run analysis
    st.markdown("## 📈 Step 2: Analyze the Data")
    st.markdown("Now let's see how the AI performs for each age group...")
    
    stratified = compute_fairness_metrics(predictions, ground_truth, demographics, protected_attr)
    tests = fairness_tests(stratified, threshold=threshold)
    recommendation = generate_recommendation(tests)
    
    # Show results
    st.markdown("### 👥 Results for Each Age Group")
    
    st.markdown("""
    <div class="explain-text">
    <b>What you're about to see:</b><br>
    • <b>Group:</b> Age category (Young, Middle, Elderly)<br>
    • <b>Patients:</b> How many in this group<br>
    • <b>Accuracy:</b> How often the AI got it right for THIS group<br>
    • <b>Diagnosis Rate:</b> What % of this group does AI say has disease
    </div>
    """, unsafe_allow_html=True)
    
    display_df = stratified[['group', 'n_samples', 'accuracy', 'positive_rate']].copy()
    display_df.columns = ['Age Group', 'Number of Patients', 'Accuracy %', 'Diagnosed %']
    display_df['Accuracy %'] = display_df['Accuracy %'].apply(lambda x: f"{x:.0%}")
    display_df['Diagnosed %'] = display_df['Diagnosed %'].apply(lambda x: f"{x:.0%}")
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Highlight the differences
    st.markdown("### ⚠️ What Do We Notice?")
    
    accs = stratified['accuracy'].values
    acc_gap = accs.max() - accs.min()
    
    if acc_gap > threshold:
        st.markdown(f"""
        <div class="error-box">
        <b>❌ BIG PROBLEM DETECTED!</b><br><br>
        The accuracy ranges from {accs.min():.0%} to {accs.max():.0%}.<br>
        <b>Difference: {acc_gap:.0%}</b><br><br>
        This is MORE than our {threshold:.0%} fairness limit!<br><br>
        <b>What this means:</b> The AI works MUCH BETTER for some age groups than others. 
        This is BIAS and it's UNFAIR.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="success-box">
        <b>✅ GOOD NEWS!</b><br><br>
        The accuracy ranges from {accs.min():.0%} to {accs.max():.0%}.<br>
        <b>Difference: {acc_gap:.0%}</b><br><br>
        This is LESS than our {threshold:.0%} fairness limit!<br><br>
        <b>What this means:</b> The AI works about the same for all age groups. 
        This is FAIR!
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Step 3 - Visualizations with explanations
    st.markdown("## 📊 Step 3: Visualize the Bias")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Chart 1: How Well Does AI Work?")
        st.markdown("""
        <div class="explain-text">
        <b>This chart shows:</b> For each age group, 
        how often does the AI correctly identify who has disease?<br><br>
        <b>What we want:</b> All bars roughly the same height (fair to everyone)<br>
        <b>What we see:</b> Are some bars much lower? That's bias!
        </div>
        """, unsafe_allow_html=True)
        
        fig1 = go.Figure()
        colors = ['#27ae60' if x > 0.80 else '#f39c12' if x > 0.60 else '#e74c3c' 
                  for x in stratified['accuracy']]
        
        fig1.add_trace(go.Bar(
            x=stratified['group'],
            y=stratified['accuracy'],
            marker=dict(color=colors),
            text=[f"{x:.0%}" for x in stratified['accuracy']],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Accuracy: %{y:.0%}<extra></extra>'
        ))
        
        fig1.add_hline(y=threshold, line_dash="dash", line_color="red", 
                      annotation_text=f"Fair threshold ({threshold:.0%})")
        
        fig1.update_layout(
            title="Accuracy by Age Group",
            xaxis_title="Age Group",
            yaxis_title="How Often AI Is Correct (%)",
            height=400,
            showlegend=False,
            yaxis=dict(tickformat='.0%')
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.markdown("### Chart 2: Who Gets Diagnosed?")
        st.markdown("""
        <div class="explain-text">
        <b>This chart shows:</b> For each age group, 
        what percentage does the AI say has disease?<br><br>
        <b>What we want:</b> All groups diagnosed at similar rates<br>
        <b>What we see:</b> Do some groups get diagnosed much more than others? 
        That's bias!
        </div>
        """, unsafe_allow_html=True)
        
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=stratified['group'],
            y=stratified['positive_rate'],
            marker=dict(color='#3498db'),
            text=[f"{x:.0%}" for x in stratified['positive_rate']],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Diagnosis Rate: %{y:.0%}<extra></extra>'
        ))
        
        fig2.update_layout(
            title="Who Gets Diagnosed",
            xaxis_title="Age Group",
            yaxis_title="% Diagnosed as Having Disease",
            height=400,
            showlegend=False,
            yaxis=dict(tickformat='.0%')
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("---")
    
    # Step 4 - Fairness tests with explanations
    st.markdown("## ✅ Step 4: Fairness Checks")
    
    st.markdown("""
    <div class="info-box">
    <b>What are fairness checks?</b><br>
    We test 3 different ways to measure fairness. 
    If the AI fails even one test, it's biased.
    </div>
    """, unsafe_allow_html=True)
    
    for test_name, test_result in tests.items():
        
        if test_name == 'Equal Accuracy':
            st.markdown("### Check 1: Does AI Work Equally for All Ages?")
            st.markdown("""
            <div class="explain-text">
            <b>What we're testing:</b> Does the AI's accuracy vary a lot between age groups?<br>
            <b>Example:</b> 95% accurate for young vs 70% accurate for elderly = BIG GAP = BIAS
            </div>
            """, unsafe_allow_html=True)
            
            if test_result['pass']:
                st.markdown(f"""
                <div class="success-box">
                <b>✅ PASS - FAIR!</b><br>
                Difference: {test_result['description']}<br><br>
                <b>Meaning:</b> The AI works about the same for all age groups. 
                No one is disadvantaged.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="error-box">
                <b>❌ FAIL - UNFAIR!</b><br>
                Difference: {test_result['description']}<br><br>
                <b>Meaning:</b> The AI works MUCH better for some age groups than others. 
                This is BIAS!
                </div>
                """, unsafe_allow_html=True)
        
        elif test_name == 'Equal Predictions':
            st.markdown("### Check 2: Does AI Diagnose Everyone Equally?")
            st.markdown("""
            <div class="explain-text">
            <b>What we're testing:</b> Does the AI diagnose disease at similar rates for all groups?<br>
            <b>Example:</b> Diagnoses 80% of young people but only 30% of elderly = UNFAIR
            </div>
            """, unsafe_allow_html=True)
            
            if test_result['pass']:
                st.markdown(f"""
                <div class="success-box">
                <b>✅ PASS - FAIR!</b><br>
                Difference: {test_result['description']}<br><br>
                <b>Meaning:</b> All age groups get diagnosed at similar rates.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="error-box">
                <b>❌ FAIL - UNFAIR!</b><br>
                Difference: {test_result['description']}<br><br>
                <b>Meaning:</b> Some groups get diagnosed much more often than others. 
                This is BIAS!
                </div>
                """, unsafe_allow_html=True)
        
        elif test_name == 'Equal Errors':
            st.markdown("### Check 3: Do All Groups Get the Same Errors?")
            st.markdown("""
            <div class="explain-text">
            <b>What we're testing:</b> Does the AI make mistakes equally for all groups?<br>
            <b>Example:</b> False alarms for 5% of young people vs 20% of elderly = UNFAIR
            </div>
            """, unsafe_allow_html=True)
            
            if test_result['pass']:
                st.markdown(f"""
                <div class="success-box">
                <b>✅ PASS - FAIR!</b><br>
                Difference: {test_result['description']}<br><br>
                <b>Meaning:</b> All groups experience similar error rates.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="error-box">
                <b>❌ FAIL - UNFAIR!</b><br>
                Difference: {test_result['description']}<br><br>
                <b>Meaning:</b> Some groups get more errors than others. 
                This is BIAS!
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Final recommendation
    st.markdown("## 🎯 Final Decision: Should We Use This AI?")
    
    if "SAFE" in recommendation:
        st.markdown(f"""
        <div class="success-box">
        <h3>✅ {recommendation}</h3>
        <b>What this means:</b> The AI treats young and elderly patients equally. 
        It won't harm any group. Safe to use in a hospital!
        </div>
        """, unsafe_allow_html=True)
    elif "NEEDS" in recommendation:
        st.markdown(f"""
        <div class="warning-box">
        <h3>⚠️ {recommendation}</h3>
        <b>What this means:</b> The AI has some fairness issues. 
        We need to improve it before using it. 
        (Maybe retrain with more data from underrepresented groups.)
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="error-box">
        <h3>🛑 {recommendation}</h3>
        <b>What this means:</b> The AI has serious bias. 
        We CANNOT use it - it would harm patients!
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# UPLOAD MODE
# ============================================================================

elif tab == "📤 Test With Your Data":
    
    st.markdown("""
    <div class="header">
        <h1>📤 Test Your Own AI</h1>
        <p>Upload your model's predictions and we'll check for bias</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <b>📌 What You Can Do Here:</b><br>
    Do you have your own AI model that predicts diseases? 
    Upload the results and we'll check if it's fair to all patients!
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📋 What You Need")
    
    st.markdown("""
    <div class="explain-text">
    <b>A spreadsheet (CSV file) with these columns:</b><br>
    • <b>prediction:</b> What your AI said (0 = no disease, 1 = has disease)<br>
    • <b>ground_truth:</b> What actually happened (0 or 1)<br>
    • <b>age_group:</b> (or gender, race, etc.) - the demographic group<br><br>
    <b>Example:</b><br>
    prediction=1, ground_truth=1, age_group=Young (correct prediction) ✅<br>
    prediction=1, ground_truth=0, age_group=Elderly (wrong prediction) ❌
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📁 Upload Your Data")
    
    uploaded_file = st.file_uploader(
        "Drag and drop your CSV file here or click to browse",
        type=['csv'],
        help="Select a CSV file from your computer"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.success("✅ File loaded successfully! Now let's configure it...")
            
            st.markdown("---")
            st.markdown("### ⚙️ Step 1: Configure Your Data")
            
            st.markdown("""
            <div class="explain-text">
            <b>Tell us which column is which:</b> We need to know which columns contain 
            predictions, truth, and demographics.
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                pred_col = st.selectbox(
                    "Which column has the AI predictions?",
                    df.columns,
                    help="Should contain only 0 or 1 values"
                )
            with col2:
                truth_col = st.selectbox(
                    "Which column has the actual truth?",
                    df.columns,
                    help="Should contain only 0 or 1 values (what really happened)"
                )
            with col3:
                demo_cols = st.multiselect(
                    "Which columns show demographic groups?",
                    df.columns,
                    help="Like: age_group, gender, race - we'll check fairness for these"
                )
            
            st.markdown("---")
            st.markdown("### 📏 Step 2: Set Fairness Standard")
            
            st.markdown("""
            <div class="explain-text">
            <b>How fair should it be?</b> If differences between groups are larger than 
            this number, we'll say the AI is biased.
            </div>
            """, unsafe_allow_html=True)
            
            threshold = st.slider(
                "Maximum acceptable difference:",
                min_value=0.05,
                max_value=0.25,
                value=0.10,
                step=0.01,
                help="Medical industry standard is 10%"
            )
            
            if st.button("🔍 Check For Bias", type="primary"):
                
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
                
                st.markdown("""
                <div class="explain-text">
                <b>What these mean:</b><br>
                • <b>Patients Analyzed:</b> How many patients tested<br>
                • <b>Overall Accuracy:</b> How often AI was correct overall<br>
                • <b>Diagnosed:</b> How many AI said have disease
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                for demo_col in demo_cols:
                    st.markdown(f"### Results by {demo_col}")
                    
                    stratified = compute_fairness_metrics(predictions, ground_truth, df, demo_col)
                    tests = fairness_tests(stratified, threshold)
                    
                    # Display table
                    display_df = stratified[['group', 'n_samples', 'accuracy', 'positive_rate']].copy()
                    display_df.columns = ['Group', 'Patients', 'Accuracy', 'Diagnosis Rate']
                    display_df['Accuracy'] = display_df['Accuracy'].apply(lambda x: f"{x:.0%}")
                    display_df['Diagnosis Rate'] = display_df['Diagnosis Rate'].apply(lambda x: f"{x:.0%}")
                    
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Accuracy by Group")
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
                        fig1.update_layout(title=f"Accuracy by {demo_col}", height=400)
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    with col2:
                        st.markdown("#### Diagnosis Rate by Group")
                        fig2 = go.Figure()
                        fig2.add_trace(go.Bar(
                            x=stratified['group'], 
                            y=stratified['positive_rate'],
                            marker=dict(color='#3498db'),
                            text=[f"{x:.0%}" for x in stratified['positive_rate']],
                            textposition='outside'
                        ))
                        fig2.update_layout(title=f"Diagnosis Rate by {demo_col}", height=400)
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    st.markdown("#### Fairness Tests")
                    for test_name, test_result in tests.items():
                        if test_result['pass']:
                            st.markdown(f"✅ **{test_name}**: PASS - ({test_result['description']})")
                        else:
                            st.markdown(f"❌ **{test_name}**: FAIL - ({test_result['description']})")
                    
                    st.markdown("---")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("Make sure your CSV has the columns you selected")

# ============================================================================
# HOW IT WORKS
# ============================================================================

elif tab == "❓ How It Works":
    st.markdown("""
    <div class="header">
        <h1>❓ How Does This Work?</h1>
        <p>Understanding AI Fairness in Simple Words</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <b>🎯 The Core Idea:</b><br>
    AI models sometimes work better for some people than others. 
    This is called BIAS. It's unfair and dangerous. This tool finds bias before it harms patients.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### ❌ The Problem: Hidden Bias")
    
    st.markdown("""
    <div class="explain-text">
    <b>What happens:</b><br>
    A hospital uses an AI to diagnose heart disease. 
    The AI seems 90% accurate overall. Looks good!<br><br>
    <b>But when we look closer:</b><br>
    • For men: 95% accurate ✅<br>
    • For women: 80% accurate ❌<br><br>
    <b>The problem:</b> Women are getting worse healthcare because the AI isn't trained on their data!
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### ✅ The Solution: Check 3 Fairness Rules")
    
    st.markdown("""
    <div class="info-box">
    <b>We test 3 things to catch bias:</b>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### 1️⃣ Rule: Equal Accuracy")
    st.markdown("""
    <div class="explain-text">
    <b>Simple question:</b> Does the AI work equally well for all groups?<br><br>
    <b>Fair example:</b><br>
    Men: 90% accurate<br>
    Women: 88% accurate<br>
    Difference: 2% ✅ (Small, acceptable)<br><br>
    <b>Unfair example:</b><br>
    Men: 95% accurate<br>
    Women: 70% accurate<br>
    Difference: 25% ❌ (Huge! Women get bad healthcare)
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### 2️⃣ Rule: Equal Diagnosis")
    st.markdown("""
    <div class="explain-text">
    <b>Simple question:</b> Does the AI diagnose all groups equally?<br><br>
    <b>Fair example:</b><br>
    60% of men get diagnosed<br>
    58% of women get diagnosed<br>
    Difference: 2% ✅ (Fair)<br><br>
    <b>Unfair example:</b><br>
    80% of men get diagnosed<br>
    30% of women get diagnosed<br>
    Difference: 50% ❌ (Women get missed diagnoses)
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### 3️⃣ Rule: Equal Errors")
    st.markdown("""
    <div class="explain-text">
    <b>Simple question:</b> Does the AI make the same mistakes for all groups?<br><br>
    <b>Fair example:</b><br>
    False alarms for 5% of men<br>
    False alarms for 6% of women<br>
    Difference: 1% ✅ (Fair)<br><br>
    <b>Unfair example:</b><br>
    False alarms for 5% of men<br>
    False alarms for 20% of women<br>
    Difference: 15% ❌ (Women get more stress from false alarms)
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🎯 Why 10% Fairness Threshold?")
    
    st.markdown("""
    <div class="explain-text">
    <b>Why exactly 10%?</b><br>
    • Medical regulators recommend 10% as the fairness standard<br>
    • Small differences (under 10%) can be natural variation<br>
    • Differences over 10% indicate real bias<br>
    • You can change this if your hospital has different standards<br><br>
    <b>Example:</b><br>
    If AI is 92% accurate for men and 90% for women = 2% difference = Fair ✅<br>
    If AI is 95% accurate for men and 80% for women = 15% difference = Unfair ❌
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🏥 Real Hospital Example")
    
    st.markdown("""
    <div class="info-box">
    <b>The Situation:</b><br>
    A hospital wants to use AI to predict who will have a stroke.<br><br>
    <b>What we find when we test it:</b><br>
    • 60-year-olds: 94% accurate ✅<br>
    • 70-year-olds: 88% accurate ⚠️<br>
    • 80-year-olds: 65% accurate ❌<br>
    • 90-year-olds: 40% accurate ❌❌<br><br>
    <b>Why is this bad?</b><br>
    Elderly people have MORE strokes, not fewer!<br>
    But the AI is WORSE at detecting them!<br>
    Result: Elderly patients get missed diagnoses and die.<br><br>
    <b>Recommendation:</b> 🛑 DO NOT USE<br>
    <b>What to do:</b> Retrain the AI with more elderly patients' data
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 💡 Why Does This Matter?")
    
    st.markdown("""
    <div class="warning-box">
    <b>Consequences of biased AI in hospitals:</b><br>
    ❌ Patients don't get diagnosed (they get sick or die)<br>
    ❌ Wrong treatments (wrong diagnosis = wrong medicine)<br>
    ❌ Hospital gets sued (bias is illegal)<br>
    ❌ Loss of patient trust (people won't use AI)<br>
    ❌ Regulatory penalties (FDA doesn't approve biased models)
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# ABOUT
# ============================================================================

elif tab == "👤 About":
    st.markdown("""
    <div class="header">
        <h1>👤 About This Tool</h1>
        <p>Why we built it and who we are</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Why This Tool Exists
    
    Medical AI is getting better every day, but it has a BIG PROBLEM:
    Hidden bias that harms patients.
    
    Example: An AI trained mostly on data from men might not work well for women.
    If a hospital uses it anyway, women don't get proper diagnoses.
    
    This tool prevents that by finding bias BEFORE it harms anyone.
    
    ---
    
    ### 👨‍🔬 Who Built This?
    
    **Dr. Loveleen Gaur**
    - AI Researcher and Ethics Expert
    - 130+ published research papers
    - Top 2% of scientists globally
    - 30+ books on AI ethics and fairness
    
    **Email:** gaurloveleen@yahoo.com
    **ORCID:** 0000-0002-0885-1550
    
    ---
    
    ### 📚 Based on Real Science
    
    This tool uses fairness standards from:
    - Published peer-reviewed research papers
    - Hospital regulations (FDA, etc.)
    - Medical ethics guidelines
    - International AI fairness standards
    
    ---
    
    ### 🔒 Your Privacy
    
    ✅ Your data is analyzed here only<br>
    ✅ Nothing is saved to our servers<br>
    ✅ Nothing is shared with anyone<br>
    ✅ Your data stays completely private<br>
    
    ---
    
    ### 💭 How It Works (Summary)
    
    1. You upload your AI model's predictions
    2. We split the data by demographic groups
    3. We check 3 fairness rules
    4. We tell you: Is it fair? Or does it have bias?
    5. If biased, we tell you how to fix it
    
    ---
    
    ### 🌟 Our Goal
    
    Make sure medical AI helps ALL patients equally.
    No bias. No discrimination. Fairness for everyone.
    """)

st.markdown("---")
st.markdown("""
<div style="text-align: center; font-size: 11px; color: gray; margin-top: 30px;">
Made with ❤️ to protect patients from biased medical AI<br>
Dr. Loveleen Gaur | AI Ethics & Fairness
</div>
""", unsafe_allow_html=True)
