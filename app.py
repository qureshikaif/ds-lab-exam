"""
Loan Risk Analytics Dashboard
Streamlit-based UI for EDA, Risk Modeling, and Predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
import os
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Loan Risk Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Outfit:wght@300;400;600;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    }
    
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif !important;
        color: #58a6ff !important;
    }
    
    .metric-card {
        background: linear-gradient(145deg, #21262d, #30363d);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        color: #58a6ff;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .risk-high {
        color: #f85149 !important;
        font-weight: bold;
    }
    
    .risk-medium {
        color: #d29922 !important;
        font-weight: bold;
    }
    
    .risk-low {
        color: #3fb950 !important;
        font-weight: bold;
    }
    
    .stSelectbox > div > div {
        background-color: #21262d;
    }
    
    .insight-box {
        background: linear-gradient(145deg, #1a2332, #21262d);
        border-left: 4px solid #58a6ff;
        padding: 15px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
    }
    
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
    }
</style>
""", unsafe_allow_html=True)


# Load data from CSV
@st.cache_data
def load_data():
    """Load loan data from CSV file"""
    csv_path = os.path.join(os.path.dirname(__file__), 'loan_data.csv')
    try:
        df = pd.read_csv(csv_path)
        return df
    except FileNotFoundError:
        st.error("loan_data.csv not found!")
        return None


@st.cache_resource
def train_model(X_train, y_train, model_type: str):
    """Train and cache the model"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if model_type == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    elif model_type == "Gradient Boosting":
        model = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5)
    else:
        model = LogisticRegression(random_state=42, max_iter=1000)
    
    model.fit(X_train_scaled, y_train)
    return model, scaler


def create_eda_charts(df: pd.DataFrame):
    """Create EDA visualizations"""
    
    # Color scheme
    colors = {
        'primary': '#58a6ff',
        'secondary': '#f0883e',
        'success': '#3fb950',
        'danger': '#f85149',
        'warning': '#d29922',
        'bg': '#0d1117',
        'card': '#21262d'
    }
    
    # 1. Distribution of Defaults
    default_counts = df['defaulted'].value_counts()
    fig_default = go.Figure(data=[
        go.Pie(
            labels=['Non-Default', 'Default'],
            values=[default_counts.get(0, 0), default_counts.get(1, 0)],
            hole=0.6,
            marker_colors=[colors['success'], colors['danger']],
            textinfo='percent+value',
            textfont=dict(size=14, family='JetBrains Mono')
        )
    ])
    fig_default.update_layout(
        title=dict(text='Default Distribution', font=dict(size=18, color=colors['primary'])),
        paper_bgcolor=colors['bg'],
        plot_bgcolor=colors['bg'],
        font=dict(color='#c9d1d9'),
        showlegend=True,
        legend=dict(orientation='h', y=-0.1)
    )
    
    # 2. Credit Score Distribution by Default Status
    fig_credit = go.Figure()
    for default_status, color, name in [(0, colors['success'], 'Non-Default'), (1, colors['danger'], 'Default')]:
        data = df[df['defaulted'] == default_status]['credit_score']
        fig_credit.add_trace(go.Histogram(
            x=data, name=name, opacity=0.7,
            marker_color=color, nbinsx=30
        ))
    fig_credit.update_layout(
        title=dict(text='Credit Score Distribution by Default Status', font=dict(size=18, color=colors['primary'])),
        xaxis_title='Credit Score',
        yaxis_title='Count',
        barmode='overlay',
        paper_bgcolor=colors['bg'],
        plot_bgcolor=colors['bg'],
        font=dict(color='#c9d1d9'),
        xaxis=dict(gridcolor='#30363d'),
        yaxis=dict(gridcolor='#30363d')
    )
    
    # 3. Age vs Loan Amount scatter
    fig_scatter = px.scatter(
        df, x='age', y='loan_amount', color='defaulted',
        color_discrete_map={0: colors['success'], 1: colors['danger']},
        opacity=0.6, size='monthly_income', size_max=20,
        labels={'defaulted': 'Default Status', 'age': 'Age', 'loan_amount': 'Loan Amount'}
    )
    fig_scatter.update_layout(
        title=dict(text='Age vs Loan Amount (size = Income)', font=dict(size=18, color=colors['primary'])),
        paper_bgcolor=colors['bg'],
        plot_bgcolor=colors['bg'],
        font=dict(color='#c9d1d9'),
        xaxis=dict(gridcolor='#30363d'),
        yaxis=dict(gridcolor='#30363d')
    )
    
    # 4. Employment Years vs Default Rate
    df['employment_bin'] = pd.cut(df['employment_years'], bins=[0, 2, 5, 10, 20, 50], labels=['0-2', '3-5', '6-10', '11-20', '20+'])
    emp_default = df.groupby('employment_bin', observed=True)['defaulted'].mean().reset_index()
    fig_emp = go.Figure(data=[
        go.Bar(
            x=emp_default['employment_bin'].astype(str),
            y=emp_default['defaulted'] * 100,
            marker_color=colors['secondary'],
            text=[f'{v:.1f}%' for v in emp_default['defaulted'] * 100],
            textposition='outside',
            textfont=dict(family='JetBrains Mono')
        )
    ])
    fig_emp.update_layout(
        title=dict(text='Default Rate by Employment Years', font=dict(size=18, color=colors['primary'])),
        xaxis_title='Employment Years',
        yaxis_title='Default Rate (%)',
        paper_bgcolor=colors['bg'],
        plot_bgcolor=colors['bg'],
        font=dict(color='#c9d1d9'),
        xaxis=dict(gridcolor='#30363d'),
        yaxis=dict(gridcolor='#30363d')
    )
    
    # 5. Correlation Heatmap
    numeric_cols = ['age', 'monthly_income', 'loan_amount', 'credit_score', 'employment_years', 'defaulted']
    corr_matrix = df[numeric_cols].corr()
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu_r',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont=dict(size=12, family='JetBrains Mono'),
        hoverongaps=False
    ))
    fig_corr.update_layout(
        title=dict(text='Feature Correlation Matrix', font=dict(size=18, color=colors['primary'])),
        paper_bgcolor=colors['bg'],
        plot_bgcolor=colors['bg'],
        font=dict(color='#c9d1d9')
    )
    
    # 6. Monthly Income Distribution
    fig_income = go.Figure()
    for default_status, color, name in [(0, colors['success'], 'Non-Default'), (1, colors['danger'], 'Default')]:
        data = df[df['defaulted'] == default_status]['monthly_income']
        fig_income.add_trace(go.Box(
            y=data, name=name, marker_color=color,
            boxmean=True
        ))
    fig_income.update_layout(
        title=dict(text='Monthly Income by Default Status', font=dict(size=18, color=colors['primary'])),
        yaxis_title='Monthly Income (₹)',
        paper_bgcolor=colors['bg'],
        plot_bgcolor=colors['bg'],
        font=dict(color='#c9d1d9'),
        yaxis=dict(gridcolor='#30363d')
    )
    
    # 7. Credit Score Bins Default Rate
    df['credit_bin'] = pd.cut(df['credit_score'], 
                               bins=[300, 500, 600, 700, 800, 850], 
                               labels=['Poor (300-500)', 'Fair (500-600)', 'Good (600-700)', 'Very Good (700-800)', 'Excellent (800-850)'])
    credit_default = df.groupby('credit_bin', observed=True)['defaulted'].agg(['mean', 'count']).reset_index()
    credit_default.columns = ['credit_bin', 'default_rate', 'count']
    
    fig_credit_bar = go.Figure(data=[
        go.Bar(
            x=credit_default['credit_bin'].astype(str),
            y=credit_default['default_rate'] * 100,
            marker_color=[colors['danger'], colors['warning'], colors['secondary'], colors['primary'], colors['success']],
            text=[f'{v:.1f}%' for v in credit_default['default_rate'] * 100],
            textposition='outside',
            textfont=dict(family='JetBrains Mono')
        )
    ])
    fig_credit_bar.update_layout(
        title=dict(text='Default Rate by Credit Score Category', font=dict(size=18, color=colors['primary'])),
        xaxis_title='Credit Score Category',
        yaxis_title='Default Rate (%)',
        paper_bgcolor=colors['bg'],
        plot_bgcolor=colors['bg'],
        font=dict(color='#c9d1d9'),
        xaxis=dict(gridcolor='#30363d', tickangle=45),
        yaxis=dict(gridcolor='#30363d')
    )
    
    # 8. Debt to Income Ratio Analysis
    df['debt_to_income'] = df['loan_amount'] / (df['monthly_income'] * 12)
    df['dti_bin'] = pd.cut(df['debt_to_income'], bins=[0, 0.5, 1, 1.5, 2, 10], labels=['<0.5', '0.5-1', '1-1.5', '1.5-2', '>2'])
    dti_default = df.groupby('dti_bin', observed=True)['defaulted'].mean().reset_index()
    
    fig_dti = go.Figure(data=[
        go.Bar(
            x=dti_default['dti_bin'].astype(str),
            y=dti_default['defaulted'] * 100,
            marker=dict(
                color=dti_default['defaulted'] * 100,
                colorscale=[[0, colors['success']], [0.5, colors['warning']], [1, colors['danger']]],
                showscale=True,
                colorbar=dict(title='Rate %')
            ),
            text=[f'{v:.1f}%' for v in dti_default['defaulted'] * 100],
            textposition='outside',
            textfont=dict(family='JetBrains Mono')
        )
    ])
    fig_dti.update_layout(
        title=dict(text='Default Rate by Debt-to-Income Ratio', font=dict(size=18, color=colors['primary'])),
        xaxis_title='Debt-to-Income Ratio',
        yaxis_title='Default Rate (%)',
        paper_bgcolor=colors['bg'],
        plot_bgcolor=colors['bg'],
        font=dict(color='#c9d1d9'),
        xaxis=dict(gridcolor='#30363d'),
        yaxis=dict(gridcolor='#30363d')
    )
    
    return {
        'default_pie': fig_default,
        'credit_hist': fig_credit,
        'scatter': fig_scatter,
        'employment': fig_emp,
        'correlation': fig_corr,
        'income_box': fig_income,
        'credit_bar': fig_credit_bar,
        'dti': fig_dti
    }


def display_model_results(y_test, y_pred, y_prob, model, feature_names):
    """Display model evaluation results"""
    
    colors = {
        'primary': '#58a6ff',
        'success': '#3fb950',
        'danger': '#f85149',
        'bg': '#0d1117'
    }
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy:.2%}")
    col2.metric("Precision", f"{precision:.2%}")
    col3.metric("Recall", f"{recall:.2%}")
    col4.metric("F1 Score", f"{f1:.2%}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted: No Default', 'Predicted: Default'],
        y=['Actual: No Default', 'Actual: Default'],
        colorscale=[[0, '#21262d'], [1, colors['primary']]],
        text=cm,
        texttemplate='%{text}',
        textfont=dict(size=20, family='JetBrains Mono'),
        hoverongaps=False
    ))
    fig_cm.update_layout(
        title=dict(text='Confusion Matrix', font=dict(size=18, color=colors['primary'])),
        paper_bgcolor=colors['bg'],
        plot_bgcolor=colors['bg'],
        font=dict(color='#c9d1d9')
    )
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=fpr, y=tpr, mode='lines',
        name=f'ROC Curve (AUC = {roc_auc:.3f})',
        line=dict(color=colors['primary'], width=2)
    ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines',
        name='Random Classifier',
        line=dict(color='#8b949e', dash='dash')
    ))
    fig_roc.update_layout(
        title=dict(text='ROC Curve', font=dict(size=18, color=colors['primary'])),
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        paper_bgcolor=colors['bg'],
        plot_bgcolor=colors['bg'],
        font=dict(color='#c9d1d9'),
        xaxis=dict(gridcolor='#30363d'),
        yaxis=dict(gridcolor='#30363d')
    )
    
    # Feature Importance
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        importance = np.abs(model.coef_[0])
    
    feat_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=True)
    
    fig_imp = go.Figure(data=[
        go.Bar(
            x=feat_imp['importance'],
            y=feat_imp['feature'],
            orientation='h',
            marker_color=colors['primary'],
            text=[f'{v:.3f}' for v in feat_imp['importance']],
            textposition='outside',
            textfont=dict(family='JetBrains Mono')
        )
    ])
    fig_imp.update_layout(
        title=dict(text='Feature Importance', font=dict(size=18, color=colors['primary'])),
        xaxis_title='Importance',
        paper_bgcolor=colors['bg'],
        plot_bgcolor=colors['bg'],
        font=dict(color='#c9d1d9'),
        xaxis=dict(gridcolor='#30363d'),
        yaxis=dict(gridcolor='#30363d')
    )
    
    return fig_cm, fig_roc, fig_imp, roc_auc


def main():
    # Header
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1 style='font-size: 2.5rem; margin-bottom: 0;'>🏦 Loan Risk Analytics Dashboard</h1>
        <p style='color: #8b949e; font-size: 1.1rem;'>Predictive Risk Modeling & Portfolio Insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("## ⚙️ Configuration")
    
    # Load data from CSV
    df = load_data()
    
    if df is None or df.empty:
        st.error("⚠️ Could not load data from loan_data.csv")
        return
    
    st.sidebar.success(f"✅ Loaded {len(df)} records from CSV")
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    # Navigation
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📊 Navigation")
    page = st.sidebar.radio(
        "Select Section",
        ["📈 Overview", "🔍 EDA & Insights", "🤖 Risk Model", "🎯 Predictions"],
        label_visibility="collapsed"
    )
    
    # Model settings in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🤖 Model Settings")
    model_type = st.sidebar.selectbox(
        "Algorithm",
        ["Random Forest", "Gradient Boosting", "Logistic Regression"]
    )
    test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2)
    
    # Prepare features
    feature_cols = ['age', 'monthly_income', 'loan_amount', 'credit_score', 'employment_years']
    X = df[feature_cols]
    y = df['defaulted']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    # ==================== OVERVIEW PAGE ====================
    if page == "📈 Overview":
        st.markdown("### 📊 Portfolio Overview")
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Customers",
                f"{len(df):,}",
                delta=None
            )
        
        with col2:
            default_rate = df['defaulted'].mean() * 100
            st.metric(
                "Default Rate",
                f"{default_rate:.1f}%",
                delta=f"{default_rate - 15:.1f}% vs benchmark" if default_rate > 15 else f"{15 - default_rate:.1f}% below benchmark"
            )
        
        with col3:
            st.metric(
                "Avg Credit Score",
                f"{df['credit_score'].mean():.0f}",
                delta=f"{df['credit_score'].mean() - 650:.0f} vs min threshold"
            )
        
        with col4:
            total_exposure = df['loan_amount'].sum()
            st.metric(
                "Total Exposure",
                f"₹{total_exposure/1e7:.1f}Cr",
                delta=None
            )
        
        st.markdown("---")
        
        # Quick insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Quick Insights")
            
            # Calculate insights
            high_risk = df[df['credit_score'] < 550]
            high_risk_default = high_risk['defaulted'].mean() * 100 if len(high_risk) > 0 else 0
            
            low_emp = df[df['employment_years'] < 2]
            low_emp_default = low_emp['defaulted'].mean() * 100 if len(low_emp) > 0 else 0
            
            st.markdown(f"""
            <div class='insight-box'>
                <strong>🔴 High Risk Segment:</strong><br>
                Customers with credit score &lt;550 have a <span class='risk-high'>{high_risk_default:.1f}%</span> default rate
            </div>
            <div class='insight-box'>
                <strong>🟡 Employment Impact:</strong><br>
                Customers with &lt;2 years employment show <span class='risk-medium'>{low_emp_default:.1f}%</span> default rate
            </div>
            <div class='insight-box'>
                <strong>🟢 Credit Score Correlation:</strong><br>
                Credit score has the <span class='risk-low'>strongest negative correlation</span> with defaults
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 📋 Data Sample")
            st.dataframe(
                df.head(10).style.background_gradient(subset=['credit_score'], cmap='RdYlGn')
                .background_gradient(subset=['defaulted'], cmap='RdYlGn_r'),
                use_container_width=True,
                height=350
            )
        
        # Distribution charts
        st.markdown("---")
        st.markdown("### 📊 Key Distributions")
        
        charts = create_eda_charts(df.copy())
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(charts['default_pie'], use_container_width=True)
        with col2:
            st.plotly_chart(charts['credit_bar'], use_container_width=True)
    
    # ==================== EDA PAGE ====================
    elif page == "🔍 EDA & Insights":
        st.markdown("### 🔍 Exploratory Data Analysis")
        
        # Summary statistics
        st.markdown("#### 📊 Summary Statistics")
        
        summary_stats = df[feature_cols + ['defaulted']].describe().T
        summary_stats['missing'] = df[feature_cols + ['defaulted']].isnull().sum()
        st.dataframe(summary_stats.style.format("{:.2f}"), use_container_width=True)
        
        st.markdown("---")
        
        # Generate all charts
        charts = create_eda_charts(df.copy())
        
        # Display charts in grid
        st.markdown("#### 📈 Visual Analysis")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Default Analysis", "Credit & Income", "Risk Factors", "Correlations"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(charts['default_pie'], use_container_width=True)
            with col2:
                st.plotly_chart(charts['credit_bar'], use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(charts['credit_hist'], use_container_width=True)
            with col2:
                st.plotly_chart(charts['income_box'], use_container_width=True)
        
        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(charts['employment'], use_container_width=True)
            with col2:
                st.plotly_chart(charts['dti'], use_container_width=True)
        
        with tab4:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(charts['correlation'], use_container_width=True)
            with col2:
                st.plotly_chart(charts['scatter'], use_container_width=True)
        
        # Key Findings
        st.markdown("---")
        st.markdown("#### 💡 Key Findings")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            corr_with_default = df[feature_cols].corrwith(df['defaulted'])
            strongest = corr_with_default.abs().idxmax()
            st.info(f"**Strongest Predictor:** {strongest} (correlation: {corr_with_default[strongest]:.3f})")
        
        with col2:
            avg_defaulter_credit = df[df['defaulted']==1]['credit_score'].mean()
            avg_non_defaulter_credit = df[df['defaulted']==0]['credit_score'].mean()
            st.warning(f"**Credit Score Gap:** Defaulters avg {avg_defaulter_credit:.0f} vs Non-defaulters {avg_non_defaulter_credit:.0f}")
        
        with col3:
            high_dti = df[df['loan_amount'] > df['monthly_income'] * 12]
            high_dti_rate = high_dti['defaulted'].mean() * 100 if len(high_dti) > 0 else 0
            st.error(f"**High DTI Risk:** {high_dti_rate:.1f}% default rate when loan > annual income")
    
    # ==================== MODEL PAGE ====================
    elif page == "🤖 Risk Model":
        st.markdown(f"### 🤖 Risk Prediction Model ({model_type})")
        
        # Train model
        with st.spinner("Training model..."):
            model, scaler = train_model(X_train, y_train, model_type)
            X_test_scaled = scaler.transform(X_test)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        st.success("Model trained successfully!")
        
        # Display results
        st.markdown("#### 📊 Model Performance")
        fig_cm, fig_roc, fig_imp, roc_auc = display_model_results(y_test, y_pred, y_prob, model, feature_cols)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_cm, use_container_width=True)
        with col2:
            st.plotly_chart(fig_roc, use_container_width=True)
        
        st.markdown("---")
        st.markdown("#### 🎯 Feature Importance")
        st.plotly_chart(fig_imp, use_container_width=True)
        
        # Model interpretation
        st.markdown("---")
        st.markdown("#### 💡 Model Insights")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            **Model Summary:**
            - Algorithm: {model_type}
            - Training samples: {len(X_train):,}
            - Test samples: {len(X_test):,}
            - ROC-AUC Score: {roc_auc:.3f}
            """)
        
        with col2:
            if hasattr(model, 'feature_importances_'):
                top_feature = feature_cols[np.argmax(model.feature_importances_)]
            else:
                top_feature = feature_cols[np.argmax(np.abs(model.coef_[0]))]
            
            st.markdown(f"""
            **Key Observations:**
            - Most important feature: **{top_feature}**
            - Model captures {roc_auc*100:.1f}% of the default signal
            - Recommended threshold: 0.5 (adjustable based on business needs)
            """)
        
        # Classification Report
        st.markdown("---")
        st.markdown("#### 📋 Detailed Classification Report")
        report = classification_report(y_test, y_pred, target_names=['Non-Default', 'Default'], output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
    
    # ==================== PREDICTIONS PAGE ====================
    elif page == "🎯 Predictions":
        st.markdown("### 🎯 Real-time Risk Predictions")
        
        # Train model for predictions
        model, scaler = train_model(X_train, y_train, model_type)
        
        tab1, tab2 = st.tabs(["Single Customer", "Batch Predictions"])
        
        with tab1:
            st.markdown("#### Enter Customer Details")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                input_age = st.number_input("Age", min_value=18, max_value=80, value=35)
                input_income = st.number_input("Monthly Income (₹)", min_value=10000, max_value=1000000, value=50000, step=5000)
            
            with col2:
                input_loan = st.number_input("Loan Amount (₹)", min_value=10000, max_value=10000000, value=500000, step=50000)
                input_credit = st.number_input("Credit Score", min_value=300, max_value=850, value=680)
            
            with col3:
                input_employment = st.number_input("Employment Years", min_value=0, max_value=50, value=5)
            
            if st.button("🔮 Predict Risk", use_container_width=True):
                # Make prediction
                input_data = np.array([[input_age, input_income, input_loan, input_credit, input_employment]])
                input_scaled = scaler.transform(input_data)
                prediction = model.predict(input_scaled)[0]
                probability = model.predict_proba(input_scaled)[0]
                
                st.markdown("---")
                st.markdown("#### 📊 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if prediction == 1:
                        st.error("### ⚠️ HIGH RISK")
                        st.markdown("Customer is likely to **DEFAULT**")
                    else:
                        st.success("### ✅ LOW RISK")
                        st.markdown("Customer is likely to **REPAY**")
                
                with col2:
                    st.metric("Default Probability", f"{probability[1]*100:.1f}%")
                    
                    # Risk gauge
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=probability[1] * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#f85149" if probability[1] > 0.5 else "#3fb950"},
                            'steps': [
                                {'range': [0, 30], 'color': "#3fb950"},
                                {'range': [30, 60], 'color': "#d29922"},
                                {'range': [60, 100], 'color': "#f85149"}
                            ],
                            'threshold': {
                                'line': {'color': "white", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    fig_gauge.update_layout(
                        paper_bgcolor='#0d1117',
                        font=dict(color='#c9d1d9'),
                        height=250
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)
                
                with col3:
                    st.markdown("**Risk Factors:**")
                    
                    # Analyze risk factors
                    risk_factors = []
                    if input_credit < 600:
                        risk_factors.append("🔴 Low credit score")
                    elif input_credit < 700:
                        risk_factors.append("🟡 Below average credit score")
                    else:
                        risk_factors.append("🟢 Good credit score")
                    
                    dti = input_loan / (input_income * 12)
                    if dti > 1.5:
                        risk_factors.append("🔴 High debt-to-income ratio")
                    elif dti > 1:
                        risk_factors.append("🟡 Moderate debt-to-income")
                    else:
                        risk_factors.append("🟢 Healthy debt-to-income")
                    
                    if input_employment < 2:
                        risk_factors.append("🔴 Limited employment history")
                    elif input_employment < 5:
                        risk_factors.append("🟡 Moderate employment tenure")
                    else:
                        risk_factors.append("🟢 Stable employment")
                    
                    for factor in risk_factors:
                        st.markdown(f"- {factor}")
        
        with tab2:
            st.markdown("#### Batch Risk Assessment")
            
            # Show predictions for test set
            X_test_scaled = scaler.transform(X_test)
            test_predictions = model.predict_proba(X_test_scaled)[:, 1]
            
            results_df = X_test.copy()
            results_df['actual_default'] = y_test.values
            results_df['predicted_probability'] = test_predictions
            results_df['predicted_default'] = (test_predictions > 0.5).astype(int)
            results_df['risk_category'] = pd.cut(
                test_predictions,
                bins=[0, 0.3, 0.6, 1],
                labels=['Low Risk', 'Medium Risk', 'High Risk']
            )
            
            # Summary by risk category
            st.markdown("##### Risk Distribution")
            risk_summary = results_df['risk_category'].value_counts()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("🟢 Low Risk", f"{risk_summary.get('Low Risk', 0):,}")
            col2.metric("🟡 Medium Risk", f"{risk_summary.get('Medium Risk', 0):,}")
            col3.metric("🔴 High Risk", f"{risk_summary.get('High Risk', 0):,}")
            
            st.markdown("---")
            
            # Display results table
            st.markdown("##### Detailed Predictions")
            
            # Add filtering
            risk_filter = st.multiselect(
                "Filter by Risk Category",
                ['Low Risk', 'Medium Risk', 'High Risk'],
                default=['High Risk', 'Medium Risk']
            )
            
            filtered_df = results_df[results_df['risk_category'].isin(risk_filter)]
            
            st.dataframe(
                filtered_df.sort_values('predicted_probability', ascending=False)
                .style.background_gradient(subset=['predicted_probability'], cmap='RdYlGn_r')
                .format({'predicted_probability': '{:.2%}'}),
                use_container_width=True,
                height=400
            )
            
            # Download button
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                "📥 Download Predictions",
                csv,
                "risk_predictions.csv",
                "text/csv",
                use_container_width=True
            )
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style='text-align: center; color: #8b949e; font-size: 0.8rem;'>
        <p>Loan Risk Analytics v1.0</p>
        <p>Built with Streamlit & Python</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
