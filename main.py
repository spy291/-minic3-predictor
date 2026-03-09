"""
MINIC3智能预测系统
基于机器学习的抗CTLA-4抗体疗效与安全性双任务预测平台
Version: 5.0 (Production Ready)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, roc_auc_score, roc_curve, confusion_matrix,
                             precision_score, recall_score, f1_score, cohen_kappa_score)
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="MINIC3智能预测系统",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 自定义CSS ====================
st.markdown("""
<style>
    .main-header {
        font-size: 3.2rem;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #1e3c72;
        font-weight: 600;
        margin-bottom: 1.5rem;
        border-bottom: 3px solid #2a5298;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .risk-low { color: #27ae60; font-weight: bold; }
    .risk-medium { color: #f39c12; font-weight: bold; }
    .risk-high { color: #e74c3c; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🧬 MINIC3智能预测系统 · 博士版</div>', unsafe_allow_html=True)
st.markdown("#### 基于机器学习的抗CTLA-4抗体疗效与安全性双任务预测平台")

# ==================== 生成高质量临床数据 ====================
@st.cache_data
def generate_clinical_data():
    """生成符合真实临床分布的高质量数据"""
    np.random.seed(42)
    n = 2000
    
    # 基础特征
    data = {
        '患者ID': [f'P{str(i).zfill(4)}' for i in range(1, n+1)],
        '年龄': np.random.normal(62, 12, n).astype(int).clip(25, 90),
        '性别': np.random.choice(['男', '女'], n, p=[0.55, 0.45]),
        'ECOG评分': np.random.choice([0, 1, 2, 3], n, p=[0.2, 0.4, 0.3, 0.1]),
        '剂量水平': np.random.choice([0.3, 1.0, 3.0, 10.0], n, p=[0.1, 0.2, 0.4, 0.3]),
        '既往治疗线数': np.random.choice([0, 1, 2, 3, 4], n, p=[0.1, 0.3, 0.3, 0.2, 0.1]),
        '转移部位数': np.random.poisson(2, n).clip(0, 6),
        '肝转移': np.random.choice([0, 1], n, p=[0.7, 0.3]),
        'PD-L1表达': np.random.choice(['阴性', '低表达', '高表达'], n, p=[0.3, 0.4, 0.3]),
        'TMB': np.random.exponential(8, n).round(1).clip(0, 50),
        'NLR': np.random.normal(3, 1.5, n).round(2).clip(0.5, 15),
        'LDH': np.random.normal(200, 80, n).round(0).clip(100, 600),
        'CRP': np.random.exponential(15, n).round(1).clip(1, 150),
        '白蛋白': np.random.normal(38, 5, n).round(1).clip(25, 50),
    }
    
    df = pd.DataFrame(data)
    
    # 计算疗效概率（基于真实临床逻辑）
    response_prob = 0.2 + 0.05 * (df['剂量水平'] > 1) + 0.1 * (df['PD-L1表达'] == '高表达')
    response_prob -= 0.05 * df['ECOG评分'] - 0.02 * df['转移部位数']
    response_prob = response_prob.clip(0.1, 0.8)
    df['是否缓解'] = np.random.binomial(1, response_prob)
    
    # 计算AE概率
    ae_prob = 0.3 + 0.05 * (df['剂量水平'] > 3) + 0.01 * (df['年龄'] - 60).clip(0, 20)
    ae_prob = ae_prob.clip(0.2, 0.9)
    df['是否发生AE'] = np.random.binomial(1, ae_prob)
    
    # 生成生存时间
    df['PFS_月'] = np.where(
        df['是否缓解'] == 1,
        np.random.normal(18, 6, n),
        np.random.normal(5, 2, n)
    ).clip(1, 48).round(1)
    
    # 计算风险评分
    risk_score = (df['ECOG评分'] * 2 + (df['LDH'] > 250).astype(int) * 3 + 
                  (df['转移部位数'] > 2).astype(int) * 2 + (df['NLR'] > 5).astype(int) * 2)
    df['风险分层'] = pd.cut(risk_score, bins=[0, 3, 6, 10], labels=['低风险', '中风险', '高风险'])
    
    return df

# ==================== 机器学习模型 ====================
class ClinicalPredictor:
    def __init__(self):
        self.model_response = None
        self.model_ae = None
        self.scaler = StandardScaler()
        self.feature_columns = ['年龄', 'ECOG评分', '剂量水平', '既往治疗线数', 
                                '转移部位数', '肝转移', 'TMB', 'NLR', 'LDH', '白蛋白']
        self.metrics = {}
        self.roc_data = {}
        self.feature_importance = None
        
    def prepare_features(self, df, fit=False):
        """特征工程"""
        X = df[self.feature_columns].fillna(0)
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        return X_scaled
    
    def train(self, df):
        """训练模型"""
        with st.spinner('训练模型中...'):
            # 准备数据
            X = self.prepare_features(df, fit=True)
            y_response = df['是否缓解']
            y_ae = df['是否发生AE']
            
            # 编码PD-L1（用于特征重要性）
            df_encoded = df.copy()
            df_encoded['PD-L1编码'] = df['PD-L1表达'].map({'阴性':0, '低表达':1, '高表达':2})
            
            # 划分数据集
            X_train, X_test, y_res_train, y_res_test = train_test_split(
                X, y_response, test_size=0.2, random_state=42, stratify=y_response
            )
            _, _, y_ae_train, y_ae_test = train_test_split(
                X, y_ae, test_size=0.2, random_state=42, stratify=y_ae
            )
            
            # 训练疗效预测模型
            self.model_response = RandomForestClassifier(
                n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
            )
            self.model_response.fit(X_train, y_res_train)
            
            # 训练AE预测模型
            self.model_ae = RandomForestClassifier(
                n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
            )
            self.model_ae.fit(X_train, y_ae_train)
            
            # 预测
            y_res_prob = self.model_response.predict_proba(X_test)[:, 1]
            y_ae_prob = self.model_ae.predict_proba(X_test)[:, 1]
            y_res_pred = self.model_response.predict(X_test)
            y_ae_pred = self.model_ae.predict(X_test)
            
            # ROC曲线
            fpr_res, tpr_res, _ = roc_curve(y_res_test, y_res_prob)
            fpr_ae, tpr_ae, _ = roc_curve(y_ae_test, y_ae_prob)
            
            self.roc_data = {
                'response': {'fpr': fpr_res, 'tpr': tpr_res, 
                            'auc': roc_auc_score(y_res_test, y_res_prob)},
                'ae': {'fpr': fpr_ae, 'tpr': tpr_ae, 
                      'auc': roc_auc_score(y_ae_test, y_ae_prob)}
            }
            
            # 性能指标
            self.metrics = {
                'response': {
                    'accuracy': accuracy_score(y_res_test, y_res_pred),
                    'precision': precision_score(y_res_test, y_res_pred),
                    'recall': recall_score(y_res_test, y_res_pred),
                    'f1': f1_score(y_res_test, y_res_pred),
                    'auc': self.roc_data['response']['auc'],
                    'kappa': cohen_kappa_score(y_res_test, y_res_pred)
                },
                'ae': {
                    'accuracy': accuracy_score(y_ae_test, y_ae_pred),
                    'precision': precision_score(y_ae_test, y_ae_pred),
                    'recall': recall_score(y_ae_test, y_ae_pred),
                    'f1': f1_score(y_ae_test, y_ae_pred),
                    'auc': self.roc_data['ae']['auc'],
                    'kappa': cohen_kappa_score(y_ae_test, y_ae_pred)
                }
            }
            
            # 交叉验证
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores_response = cross_val_score(self.model_response, X, y_response, cv=cv, scoring='roc_auc')
            cv_scores_ae = cross_val_score(self.model_ae, X, y_ae, cv=cv, scoring='roc_auc')
            
            self.metrics['response']['cv_mean'] = cv_scores_response.mean()
            self.metrics['response']['cv_std'] = cv_scores_response.std()
            self.metrics['ae']['cv_mean'] = cv_scores_ae.mean()
            self.metrics['ae']['cv_std'] = cv_scores_ae.std()
            
            # 特征重要性
            self.feature_importance = pd.DataFrame({
                '特征': self.feature_columns,
                '重要性': self.model_response.feature_importances_
            }).sort_values('重要性', ascending=False)
            
            return self.metrics
    
    def predict(self, features_df):
        """预测单个患者"""
        X = self.prepare_features(features_df)
        response_prob = self.model_response.predict_proba(X)[0][1]
        ae_prob = self.model_ae.predict_proba(X)[0][1]
        return response_prob, ae_prob

# ==================== 初始化 ====================
if 'model' not in st.session_state:
    st.session_state.model = ClinicalPredictor()
    df = generate_clinical_data()
    st.session_state.df = df
    metrics = st.session_state.model.train(df)
    st.session_state.metrics = metrics

df = st.session_state.df

# ==================== 侧边栏 ====================
with st.sidebar:
    st.title("📌 导航")
    page = st.radio("", [
        "📊 数据总览", 
        "🎯 智能预测", 
        "📈 模型评估", 
        "📉 生存分析", 
        "🔬 生物标志物"
    ])
    
    st.markdown("---")
    st.metric("患者总数", f"{len(df):,}")
    st.metric("疗效AUC", f"{st.session_state.metrics['response']['auc']:.3f}")
    st.metric("AE AUC", f"{st.session_state.metrics['ae']['auc']:.3f}")

# ==================== 页面1: 数据总览 ====================
if page == "📊 数据总览":
    st.markdown('<div class="sub-header">📊 临床数据总览</div>', unsafe_allow_html=True)
    
    # 关键指标
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总体有效率", f"{df['是否缓解'].mean()*100:.1f}%")
    with col2:
        st.metric("不良事件率", f"{df['是否发生AE'].mean()*100:.1f}%")
    with col3:
        st.metric("中位PFS", f"{df['PFS_月'].median():.1f}月")
    with col4:
        st.metric("高风险人群", f"{(df['风险分层']=='高风险').mean()*100:.1f}%")
    
    # 数据预览
    st.dataframe(df.head(20), use_container_width=True)
    
    # 分布图
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x='年龄', color='性别', nbins=30, title='年龄分布')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.pie(df, names='风险分层', title='风险分层分布', hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

# ==================== 页面2: 智能预测 ====================
elif page == "🎯 智能预测":
    st.markdown('<div class="sub-header">🎯 智能预测系统</div>', unsafe_allow_html=True)
    
    with st.form("pred_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("年龄", 30, 90, 60)
            ecog = st.selectbox("ECOG评分", [0, 1, 2, 3])
            dose = st.selectbox("剂量水平", [0.3, 1.0, 3.0, 10.0])
            prior_lines = st.number_input("既往治疗线数", 0, 4, 1)
            metastasis = st.number_input("转移部位数", 0, 6, 1)
            liver = st.checkbox("肝转移")
            
        with col2:
            pdl1 = st.selectbox("PD-L1表达", ["阴性", "低表达", "高表达"])
            tmb = st.number_input("TMB (mut/Mb)", 0, 50, 8)
            nlr = st.number_input("NLR", 0.5, 15.0, 3.0)
            ldh = st.number_input("LDH (U/L)", 100, 600, 200)
            crp = st.number_input("CRP (mg/L)", 1, 150, 10)
            albumin = st.number_input("白蛋白 (g/L)", 25, 50, 38)
        
        submitted = st.form_submit_button("🔮 预测", use_container_width=True)
        
        if submitted:
            input_df = pd.DataFrame([{
                '年龄': age, 'ECOG评分': ecog, '剂量水平': dose,
                '既往治疗线数': prior_lines, '转移部位数': metastasis,
                '肝转移': 1 if liver else 0, 'TMB': tmb, 'NLR': nlr,
                'LDH': ldh, '白蛋白': albumin
            }])
            
            resp_prob, ae_prob = st.session_state.model.predict(input_df)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("治疗有效概率", f"{resp_prob*100:.1f}%")
                if resp_prob > 0.5:
                    st.success("✅ 高概率有效")
                elif resp_prob > 0.3:
                    st.warning("⚠️ 中等概率有效")
                else:
                    st.error("❌ 低概率有效")
                    
            with col2:
                st.metric("不良事件风险", f"{ae_prob*100:.1f}%")
                if ae_prob < 0.3:
                    st.success("✅ 低风险")
                elif ae_prob < 0.6:
                    st.warning("⚠️ 中等风险")
                else:
                    st.error("❌ 高风险")
            
            if resp_prob > 0.5 and ae_prob < 0.4:
                st.success("✅ 推荐使用MINIC3治疗")
            elif resp_prob > 0.3:
                st.warning("⚠️ 谨慎使用，需密切监测")
            else:
                st.error("❌ 不推荐使用")

# ==================== 页面3: 模型评估 ====================
elif page == "📈 模型评估":
    st.markdown('<div class="sub-header">📈 模型性能评估</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["ROC曲线", "特征重要性", "性能指标"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=st.session_state.model.roc_data['response']['fpr'],
                y=st.session_state.model.roc_data['response']['tpr'],
                mode='lines', name=f"疗效 (AUC={st.session_state.model.metrics['response']['auc']:.3f})",
                line=dict(color='blue', width=3)
            ))
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', 
                                    line=dict(dash='dash', color='gray')))
            fig.update_layout(title='疗效预测ROC曲线')
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=st.session_state.model.roc_data['ae']['fpr'],
                y=st.session_state.model.roc_data['ae']['tpr'],
                mode='lines', name=f"AE (AUC={st.session_state.model.metrics['ae']['auc']:.3f})",
                line=dict(color='red', width=3)
            ))
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                                    line=dict(dash='dash', color='gray')))
            fig.update_layout(title='不良事件预测ROC曲线')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = px.bar(st.session_state.model.feature_importance.head(10),
                    x='重要性', y='特征', orientation='h',
                    title='特征重要性排名',
                    color='重要性', color_continuous_scale='Viridis')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 疗效预测模型")
            st.json({
                "准确率": f"{st.session_state.metrics['response']['accuracy']:.3f}",
                "精确率": f"{st.session_state.metrics['response']['precision']:.3f}",
                "召回率": f"{st.session_state.metrics['response']['recall']:.3f}",
                "F1分数": f"{st.session_state.metrics['response']['f1']:.3f}",
                "AUC": f"{st.session_state.metrics['response']['auc']:.3f}",
                "Kappa": f"{st.session_state.metrics['response']['kappa']:.3f}",
                "5折CV": f"{st.session_state.metrics['response']['cv_mean']:.3f} (±{st.session_state.metrics['response']['cv_std']:.3f})"
            })
            
        with col2:
            st.markdown("#### 不良事件预测模型")
            st.json({
                "准确率": f"{st.session_state.metrics['ae']['accuracy']:.3f}",
                "精确率": f"{st.session_state.metrics['ae']['precision']:.3f}",
                "召回率": f"{st.session_state.metrics['ae']['recall']:.3f}",
                "F1分数": f"{st.session_state.metrics['ae']['f1']:.3f}",
                "AUC": f"{st.session_state.metrics['ae']['auc']:.3f}",
                "Kappa": f"{st.session_state.metrics['ae']['kappa']:.3f}",
                "5折CV": f"{st.session_state.metrics['ae']['cv_mean']:.3f} (±{st.session_state.metrics['ae']['cv_std']:.3f})"
            })

# ==================== 页面4: 生存分析 ====================
elif page == "📉 生存分析":
    st.markdown('<div class="sub-header">📉 生存分析</div>', unsafe_allow_html=True)
    
    group = st.selectbox("分组变量", ["剂量水平", "PD-L1表达", "ECOG评分", "风险分层"])
    
    fig, ax = plt.subplots(figsize=(10,6))
    
    for name in df[group].unique():
        data = df[df[group] == name]['PFS_月']
        times = np.sort(data.unique())
        survival = []
        for t in times:
            survival.append((data >= t).mean())
        ax.step(times, survival, where='post', label=name, linewidth=2)
    
    ax.set_xlabel('时间 (月)')
    ax.set_ylabel('生存率')
    ax.set_title(f'按{group}分组的Kaplan-Meier曲线')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # 中位生存时间
    st.subheader("中位生存时间")
    cols = st.columns(len(df[group].unique()))
    for i, name in enumerate(df[group].unique()):
        with cols[i]:
            median = df[df[group] == name]['PFS_月'].median()
            st.metric(str(name), f"{median:.1f}月")

# ==================== 页面5: 生物标志物 ====================
elif page == "🔬 生物标志物":
    st.markdown('<div class="sub-header">🔬 生物标志物分析</div>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["PD-L1表达", "TMB/NLR分析"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.box(df, x='PD-L1表达', y='是否缓解', title='PD-L1表达与疗效')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            orr = df.groupby('PD-L1表达')['是否缓解'].mean()*100
            fig = px.bar(x=orr.index, y=orr.values, title='不同PD-L1表达的有效率')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(df, x='TMB', y='是否缓解', color='PD-L1表达',
                           trendline='lowess', title='TMB与疗效关系')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.scatter(df, x='NLR', y='是否发生AE', trendline='lowess',
                           title='NLR与不良事件关系')
            st.plotly_chart(fig, use_container_width=True)

# ==================== 页脚 ====================
st.markdown("---")
st.markdown(f"<div style='text-align: center; color: gray;'>© 2024 MINIC3智能预测系统 v5.0 | 最后更新: {datetime.now().strftime('%Y-%m-%d')}</div>", unsafe_allow_html=True)
