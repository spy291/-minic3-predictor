"""
MINIC3智能预测系统
基于机器学习的免疫治疗疗效与安全性双任务预测平台
Version: 4.1 
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import (accuracy_score, roc_auc_score, roc_curve, confusion_matrix,
                             precision_score, recall_score, f1_score, classification_report,
                             cohen_kappa_score, matthews_corrcoef, calibration_curve)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# 页面配置
st.set_page_config(
    page_title="MINIC3智能预测系统",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        margin-bottom: 1rem;
        text-align: center;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2d3748;
        font-weight: 600;
        margin-bottom: 1.5rem;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .info-card {
        background: #f7fafc;
        padding: 1.5rem;
        border-radius: 1rem;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    .risk-low { color: #48bb78; font-weight: 700; }
    .risk-medium { color: #ecc94b; font-weight: 700; }
    .risk-high { color: #f56565; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🧬 MINIC3智能预测系统 · 专业版</div>', unsafe_allow_html=True)
st.markdown("#### 基于机器学习的抗CTLA-4抗体疗效与安全性双任务预测平台")

# 生成高质量模拟数据
@st.cache_data
def generate_high_quality_data():
    np.random.seed(42)
    n_patients = 1000
    
    data = {
        '患者ID': [f'P{str(i).zfill(4)}' for i in range(1, n_patients + 1)],
        '年龄': np.random.normal(62, 12, n_patients).astype(int).clip(25, 90),
        '性别': np.random.choice(['男', '女'], n_patients, p=[0.55, 0.45]),
        'ECOG评分': np.random.choice([0, 1, 2, 3], n_patients, p=[0.2, 0.4, 0.3, 0.1]),
        '剂量水平(mg/kg)': np.random.choice([0.3, 1.0, 3.0, 10.0], n_patients, p=[0.1, 0.2, 0.4, 0.3]),
        '既往治疗线数': np.random.choice([0, 1, 2, 3], n_patients, p=[0.2, 0.4, 0.3, 0.1]),
        '转移部位数': np.random.poisson(2, n_patients).clip(0, 5),
        '肝转移': np.random.choice([0, 1], n_patients, p=[0.7, 0.3]),
        'PD-L1表达': np.random.choice(['阴性', '低表达', '高表达'], n_patients, p=[0.3, 0.4, 0.3]),
        'TMB(mut/Mb)': np.random.exponential(8, n_patients).round(1).clip(0, 50),
        'NLR': np.random.normal(3, 1.5, n_patients).round(2).clip(0.5, 15),
        'LDH(U/L)': np.random.normal(200, 80, n_patients).round(0).clip(100, 600),
        'CRP(mg/L)': np.random.exponential(15, n_patients).round(1).clip(1, 150),
        '白蛋白(g/L)': np.random.normal(38, 5, n_patients).round(1).clip(25, 50),
    }
    
    df = pd.DataFrame(data)
    
    # 生成疗效
    response_prob = 0.2 + 0.05 * (df['剂量水平(mg/kg)'] > 1) + 0.1 * (df['PD-L1表达'] == '高表达')
    response_prob = response_prob.clip(0.1, 0.8)
    df['是否缓解'] = np.random.binomial(1, response_prob)
    
    # 生成AE
    ae_prob = 0.3 + 0.05 * (df['剂量水平(mg/kg)'] > 3) + 0.01 * (df['年龄'] - 60).clip(0, 20)
    ae_prob = ae_prob.clip(0.2, 0.9)
    df['是否发生AE'] = np.random.binomial(1, ae_prob)
    
    # 生成PFS
    df['PFS_月'] = np.where(
        df['是否缓解'] == 1,
        np.random.normal(15, 5, len(df)),
        np.random.normal(5, 2, len(df))
    ).clip(1, 36).round(1)
    
    df['肿瘤缓解状态'] = np.where(df['是否缓解'] == 1, '有效', '无效')
    df['风险分层'] = pd.cut(df['ECOG评分'] + (df['LDH(U/L)'] > 250).astype(int) * 2, 
                           bins=[0, 2, 4, 6], labels=['低风险', '中风险', '高风险'])
    
    return df

# 高级机器学习模型
class AdvancedClinicalPredictor:
    def __init__(self):
        self.model_response = None
        self.model_ae = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.importance_df = None
        self.roc_data = None
        self.cv_scores = None
        self.metrics = None
        self.calibration_data = None
        self.label_encoders = {}
        
    def prepare_features(self, df, fit_scaler=False):
        feature_df = df.copy()
        
        # 编码分类变量
        feature_df['性别编码'] = feature_df['性别'].map({'男': 0, '女': 1})
        feature_df['PD-L1编码'] = feature_df['PD-L1表达'].map({'阴性': 0, '低表达': 1, '高表达': 2})
        
        self.feature_columns = [
            '年龄', 'ECOG评分', '剂量水平(mg/kg)', '既往治疗线数',
            '转移部位数', '肝转移', 'PD-L1编码', 'TMB(mut/Mb)',
            'NLR', 'LDH(U/L)', 'CRP(mg/L)', '白蛋白(g/L)'
        ]
        
        X = feature_df[self.feature_columns].fillna(0)
        
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
            
        return pd.DataFrame(X_scaled, columns=self.feature_columns)
    
    def _bootstrap_auc(self, y_true, y_prob, n_bootstrap=1000):
        """Bootstrap法计算AUC置信区间 - 修复版"""
        aucs = []
        n = len(y_true)
        
        # 转换为numpy数组避免索引问题
        if hasattr(y_true, 'values'):
            y_true = y_true.values
        if hasattr(y_prob, 'values'):
            y_prob = y_prob.values
        
        for _ in range(n_bootstrap):
            idx = np.random.choice(n, n, replace=True)
            if len(np.unique(y_true[idx])) > 1:
                try:
                    aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
                except:
                    continue
        
        if len(aucs) > 0:
            return [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]
        else:
            return [0.0, 0.0]
    
    def train(self, df):
        with st.spinner('正在训练模型...'):
            X = self.prepare_features(df, fit_scaler=True)
            y_response = df['是否缓解']
            y_ae = df['是否发生AE']
            
            X_train, X_test, y_response_train, y_response_test = train_test_split(
                X, y_response, test_size=0.2, random_state=42, stratify=y_response
            )
            X_train_ae, X_test_ae, y_ae_train, y_ae_test = train_test_split(
                X, y_ae, test_size=0.2, random_state=42, stratify=y_ae
            )
            
            self.model_response = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
            self.model_response.fit(X_train, y_response_train)
            
            self.model_ae = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
            self.model_ae.fit(X_train_ae, y_ae_train)
            
            # 预测概率
            y_response_prob = self.model_response.predict_proba(X_test)[:, 1]
            y_ae_prob = self.model_ae.predict_proba(X_test_ae)[:, 1]
            
            # ROC曲线
            fpr_res, tpr_res, _ = roc_curve(y_response_test, y_response_prob)
            fpr_ae, tpr_ae, _ = roc_curve(y_ae_test, y_ae_prob)
            
            # 修复这里：调用修复后的_bootstrap_auc方法
            self.roc_data = {
                'response': {
                    'fpr': fpr_res, 'tpr': tpr_res,
                    'auc': roc_auc_score(y_response_test, y_response_prob),
                    'auc_ci': self._bootstrap_auc(y_response_test, y_response_prob)
                },
                'ae': {
                    'fpr': fpr_ae, 'tpr': tpr_ae,
                    'auc': roc_auc_score(y_ae_test, y_ae_prob),
                    'auc_ci': self._bootstrap_auc(y_ae_test, y_ae_prob)
                }
            }
            
            # 校准曲线
            prob_true_res, prob_pred_res = calibration_curve(y_response_test, y_response_prob, n_bins=10)
            prob_true_ae, prob_pred_ae = calibration_curve(y_ae_test, y_ae_prob, n_bins=10)
            
            self.calibration_data = {
                'response': {'prob_true': prob_true_res, 'prob_pred': prob_pred_res},
                'ae': {'prob_true': prob_true_ae, 'prob_pred': prob_pred_ae}
            }
            
            # 特征重要性
            self.importance_df = pd.DataFrame({
                '特征': self.feature_columns,
                '重要性': self.model_response.feature_importances_
            }).sort_values('重要性', ascending=False)
            
            # 交叉验证
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            self.cv_scores = {
                'response': cross_val_score(self.model_response, X, y_response, cv=cv, scoring='roc_auc'),
                'ae': cross_val_score(self.model_ae, X, y_ae, cv=cv, scoring='roc_auc')
            }
            
            y_response_pred = self.model_response.predict(X_test)
            y_ae_pred = self.model_ae.predict(X_test_ae)
            
            self.metrics = {
                'response': {
                    'accuracy': accuracy_score(y_response_test, y_response_pred),
                    'precision': precision_score(y_response_test, y_response_pred),
                    'recall': recall_score(y_response_test, y_response_pred),
                    'f1': f1_score(y_response_test, y_response_pred),
                    'auc': self.roc_data['response']['auc']
                },
                'ae': {
                    'accuracy': accuracy_score(y_ae_test, y_ae_pred),
                    'precision': precision_score(y_ae_test, y_ae_pred),
                    'recall': recall_score(y_ae_test, y_ae_pred),
                    'f1': f1_score(y_ae_test, y_ae_pred),
                    'auc': self.roc_data['ae']['auc']
                }
            }
            
            return self.metrics
    
    def predict_patient(self, patient_features):
        features_scaled = self.scaler.transform(patient_features)
        response_prob = self.model_response.predict_proba(features_scaled)[0][1]
        ae_prob = self.model_ae.predict_proba(features_scaled)[0][1]
        
        return {
            'response_prob': response_prob,
            'ae_prob': ae_prob
        }

# 初始化
if 'model' not in st.session_state:
    st.session_state.model = AdvancedClinicalPredictor()
    with st.spinner('正在生成数据并训练模型...'):
        df = generate_high_quality_data()
        st.session_state.df = df
        metrics = st.session_state.model.train(df)
        st.session_state.metrics = metrics

df = st.session_state.df

# 侧边栏导航
with st.sidebar:
    st.title("导航菜单")
    page = st.radio("选择功能", ["数据总览", "智能预测", "模型分析", "生存分析"])
    
    st.markdown("---")
    st.metric("患者总数", f"{len(df):,}")
    st.metric("疗效AUC", f"{st.session_state.metrics['response']['auc']:.3f}")

# 数据总览页面
if page == "数据总览":
    st.subheader("📊 数据总览")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总体有效率", f"{df['是否缓解'].mean()*100:.1f}%")
    with col2:
        st.metric("不良事件率", f"{df['是否发生AE'].mean()*100:.1f}%")
    with col3:
        st.metric("中位PFS", f"{df['PFS_月'].median():.1f} 月")
    with col4:
        st.metric("高风险人群", f"{(df['风险分层']=='高风险').mean()*100:.1f}%")
    
    st.dataframe(df.head(20))

# 智能预测页面
elif page == "智能预测":
    st.subheader("🎯 智能预测")
    
    with st.form("pred_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("年龄", 30, 90, 60)
            dose = st.selectbox("剂量", [0.3, 1.0, 3.0, 10.0])
            ecog = st.selectbox("ECOG", [0, 1, 2, 3])
            pdl1 = st.selectbox("PD-L1", ["阴性", "低表达", "高表达"])
        
        with col2:
            tmb = st.number_input("TMB", 0, 50, 8)
            nlr = st.number_input("NLR", 0.5, 15.0, 3.0)
            ldh = st.number_input("LDH", 100, 600, 200)
            crp = st.number_input("CRP", 1, 150, 10)
        
        submitted = st.form_submit_button("预测", use_container_width=True)
        
        if submitted:
            input_data = pd.DataFrame([{
                '年龄': age, '性别': '男', 'ECOG评分': ecog,
                '剂量水平(mg/kg)': dose, '既往治疗线数': 1,
                '转移部位数': 1, '肝转移': 0,
                'PD-L1表达': pdl1, 'TMB(mut/Mb)': tmb,
                'NLR': nlr, 'LDH(U/L)': ldh,
                'CRP(mg/L)': crp, '白蛋白(g/L)': 38
            }])
            
            input_data['性别编码'] = 0
            input_data['PD-L1编码'] = {'阴性':0, '低表达':1, '高表达':2}[pdl1]
            
            features = st.session_state.model.prepare_features(input_data)
            pred = st.session_state.model.predict_patient(features)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("有效概率", f"{pred['response_prob']*100:.1f}%")
            with col2:
                st.metric("AE风险", f"{pred['ae_prob']*100:.1f}%")
            
            if pred['response_prob'] > 0.5 and pred['ae_prob'] < 0.4:
                st.success("✅ 推荐使用")
            elif pred['response_prob'] > 0.3:
                st.warning("⚠️ 谨慎使用")
            else:
                st.error("❌ 不推荐")

# 模型分析页面
elif page == "模型分析":
    st.subheader("📊 模型分析")
    
    tab1, tab2, tab3 = st.tabs(["ROC曲线", "特征重要性", "性能指标"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.line(x=st.session_state.model.roc_data['response']['fpr'],
                         y=st.session_state.model.roc_data['response']['tpr'],
                         title=f"疗效预测 ROC (AUC={st.session_state.model.roc_data['response']['auc']:.3f})")
            fig.add_scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash'))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.line(x=st.session_state.model.roc_data['ae']['fpr'],
                         y=st.session_state.model.roc_data['ae']['tpr'],
                         title=f"AE预测 ROC (AUC={st.session_state.model.roc_data['ae']['auc']:.3f})")
            fig.add_scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash'))
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = px.bar(st.session_state.model.importance_df.head(10),
                    x='重要性', y='特征', orientation='h',
                    title='特征重要性排名')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("疗效准确率", f"{st.session_state.metrics['response']['accuracy']:.3f}")
            st.metric("疗效AUC", f"{st.session_state.metrics['response']['auc']:.3f}")
        
        with col2:
            st.metric("AE准确率", f"{st.session_state.metrics['ae']['accuracy']:.3f}")
            st.metric("AEAUC", f"{st.session_state.metrics['ae']['auc']:.3f}")

# 生存分析页面
elif page == "生存分析":
    st.subheader("📈 生存分析")
    
    fig, ax = plt.subplots(figsize=(10,6))
    
    for dose in sorted(df['剂量水平(mg/kg)'].unique()):
        dose_data = df[df['剂量水平(mg/kg)'] == dose]
        time_points = np.sort(dose_data['PFS_月'].unique())
        survival = []
        
        for t in time_points:
            survival.append((dose_data['PFS_月'] >= t).mean())
        
        ax.step(time_points, survival, where='post', label=f'{dose} mg/kg')
    
    ax.set_xlabel('时间 (月)')
    ax.set_ylabel('生存率')
    ax.set_title('Kaplan-Meier生存曲线')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
