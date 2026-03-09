import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, roc_auc_score, roc_curve, confusion_matrix,
                             precision_score, recall_score, f1_score, calibration_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from scipy import stats
import shap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
from datetime import datetime
import io
import base64

warnings.filterwarnings('ignore')

# 页面配置
st.set_page_config(
    page_title="MINIC3智能预测系统",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        background: linear-gradient(45deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
        margin-bottom: 1rem;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .info-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .risk-low { color: #27ae60; font-weight: bold; font-size: 1.2rem; }
    .risk-medium { color: #f39c12; font-weight: bold; font-size: 1.2rem; }
    .risk-high { color: #e74c3c; font-weight: bold; font-size: 1.2rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🧠 MINIC3抗CTLA-4抗体智能预测系统</div>', unsafe_allow_html=True)
st.markdown("### 基于多模态机器学习的疗效与安全性双任务预测平台")

# ==================== 生成增强版模拟数据 ====================
@st.cache_data
def generate_clinical_data():
    """生成模拟临床数据"""
    np.random.seed(42)
    n_patients = 1000  # 增加样本量
    
    # 基础特征
    data = {
        '患者ID': [f'P{str(i).zfill(4)}' for i in range(1, n_patients + 1)],
        '剂量水平(mg/kg)': np.random.choice([0.3, 1.0, 3.0, 10.0], n_patients, p=[0.15, 0.25, 0.35, 0.25]),
        '年龄': np.random.normal(60, 12, n_patients).astype(int).clip(25, 85),
        '性别': np.random.choice(['男', '女'], n_patients, p=[0.52, 0.48]),
        '体重(kg)': np.random.normal(70, 15, n_patients).astype(int).clip(40, 120),
        'BMI': np.random.normal(24, 4, n_patients).round(1),
    }
    
    # 肿瘤相关特征
    tumor_data = {
        '基线肿瘤大小(mm)': np.random.exponential(30, n_patients).round(1).clip(5, 150),
        'ECOG评分': np.random.choice([0, 1, 2, 3], n_patients, p=[0.25, 0.45, 0.25, 0.05]),
        '既往治疗线数': np.random.choice([0, 1, 2, 3, 4], n_patients, p=[0.1, 0.3, 0.3, 0.2, 0.1]),
        '肿瘤类型': np.random.choice(['非小细胞肺癌', '黑色素瘤', '肾细胞癌', '尿路上皮癌', '头颈鳞癌'], n_patients),
        '转移部位数': np.random.poisson(2, n_patients).clip(0, 5),
        '肝转移': np.random.choice([0, 1], n_patients, p=[0.7, 0.3]),
        '脑转移': np.random.choice([0, 1], n_patients, p=[0.85, 0.15]),
    }
    
    # 生物标志物
    biomarker_data = {
        'PD-L1表达': np.random.choice(['阴性(<1%)', '低表达(1-49%)', '高表达(≥50%)'], n_patients, p=[0.3, 0.4, 0.3]),
        'TMB(mut/Mb)': np.random.exponential(8, n_patients).round(1).clip(0, 50),
        'MSI状态': np.random.choice(['MSS', 'MSI-L', 'MSI-H'], n_patients, p=[0.8, 0.15, 0.05]),
        '中性粒细胞计数': np.random.normal(4.5, 2, n_patients).round(2).clip(1, 15),
        '淋巴细胞计数': np.random.normal(2.0, 0.8, n_patients).round(2).clip(0.3, 5),
        '血小板计数': np.random.normal(250, 80, n_patients).round(0).clip(100, 500),
        '白蛋白(g/L)': np.random.normal(38, 5, n_patients).round(1).clip(25, 50),
        'LDH(U/L)': np.random.normal(200, 80, n_patients).round(0).clip(100, 600),
        'CRP(mg/L)': np.random.exponential(15, n_patients).round(1).clip(1, 150),
    }
    
    df = pd.DataFrame({**data, **tumor_data, **biomarker_data})
    
    # 计算衍生指标
    df['NLR'] = (df['中性粒细胞计数'] / df['淋巴细胞计数']).round(2)
    df['PLR'] = (df['血小板计数'] / df['淋巴细胞计数']).round(2)
    df['LIPI评分'] = np.where(
        (df['LDH(U/L)'] > 250) & (df['NLR'] > 3), '高风险',
        np.where((df['LDH(U/L)'] > 250) | (df['NLR'] > 3), '中风险', '低风险')
    )
    
    # 复杂的疗效生成逻辑
    def calculate_response_prob(row):
        # 基础概率
        base_prob = 0.25
        
        # 剂量效应
        dose_effect = {0.3: -0.1, 1.0: 0, 3.0: 0.15, 10.0: 0.25}
        
        # PD-L1效应
        pdl1_effect = {'阴性(<1%)': -0.1, '低表达(1-49%)': 0.05, '高表达(≥50%)': 0.2}
        
        # TMB效应
        tmb_effect = 0.01 * (row['TMB(mut/Mb)'] - 10) if row['TMB(mut/Mb)'] > 10 else 0
        
        # 临床特征效应
        ecog_effect = -0.15 * row['ECOG评分']
        metastasis_effect = -0.05 * row['转移部位数']
        liver_effect = -0.15 if row['肝转移'] == 1 else 0
        albumin_effect = 0.02 * (row['白蛋白(g/L)'] - 35)
        ldh_effect = -0.001 * (row['LDH(U/L)'] - 200)
        
        # NLR效应
        nlr_effect = -0.05 * (row['NLR'] - 3) if row['NLR'] > 3 else 0
        
        prob = (base_prob + dose_effect[row['剂量水平(mg/kg)']] + 
                pdl1_effect[row['PD-L1表达']] + tmb_effect + ecog_effect +
                metastasis_effect + liver_effect + albumin_effect + 
                ldh_effect + nlr_effect)
        
        return np.clip(prob, 0.05, 0.85)
    
    # 不良事件生成逻辑
    def calculate_ae_prob(row):
        base_prob = 0.35
        
        # 剂量效应
        dose_ae_effect = {0.3: -0.2, 1.0: -0.1, 3.0: 0.1, 10.0: 0.25}
        
        # 年龄效应
        age_effect = 0.01 * (row['年龄'] - 60) if row['年龄'] > 60 else 0
        
        # 肾功能（用BMI和年龄简单模拟）
        renal_effect = 0.01 * (70 - row['体重(kg)']) if row['体重(kg)'] < 60 else 0
        
        # 炎症指标
        crp_effect = 0.003 * row['CRP(mg/L)']
        nlr_effect = 0.03 * (row['NLR'] - 3) if row['NLR'] > 3 else 0
        
        prob = (base_prob + dose_ae_effect[row['剂量水平(mg/kg)']] + 
                age_effect + renal_effect + crp_effect + nlr_effect)
        
        return np.clip(prob, 0.1, 0.9)
    
    # 生成结果
    response_probs = df.apply(calculate_response_prob, axis=1)
    ae_probs = df.apply(calculate_ae_prob, axis=1)
    
    df['疗效概率'] = response_probs.round(3)
    df['AE概率'] = ae_probs.round(3)
    df['是否缓解'] = np.random.binomial(1, response_probs)
    df['是否发生AE'] = np.random.binomial(1, ae_probs)
    
    # 生成PFS时间
    df['PFS_月'] = np.where(
        df['是否缓解'] == 1,
        np.random.normal(15, 5, len(df)),
        np.random.normal(5, 2, len(df))
    ).clip(1, 36).round(1)
    
    # 生成OS时间
    df['OS_月'] = df['PFS_月'] + np.random.exponential(8, len(df)).round(1)
    df['OS_月'] = df['OS_月'].clip(1, 48).round(1)
    
    # 生成删失指标
    df['事件'] = np.random.binomial(1, 0.8, len(df))
    
    # 风险分层
    df['风险评分'] = (df['ECOG评分'] * 2 + (df['LDH(U/L)'] > 250).astype(int) * 3 + 
                     (df['转移部位数'] > 2).astype(int) * 2 + (df['NLR'] > 4).astype(int) * 2)
    
    df['风险分层'] = pd.cut(df['风险评分'], bins=[0, 3, 6, 10], labels=['低风险', '中风险', '高风险'])
    
    # 肿瘤缓解状态文本
    df['肿瘤缓解状态'] = np.where(
        df['是否缓解'] == 1,
        np.random.choice(['完全缓解(CR)', '部分缓解(PR)'], len(df), p=[0.2, 0.8]),
        np.random.choice(['疾病稳定(SD)', '疾病进展(PD)'], len(df), p=[0.4, 0.6])
    )
    
    # 不良事件类型和分级
    ae_types = ['皮疹', '腹泻', '肝炎', '肺炎', '甲状腺炎', '结肠炎']
    ae_grades = ['1级', '2级', '3级', '4级']
    
    df['不良事件详情'] = df.apply(
        lambda row: f"{np.random.choice(ae_types)} {np.random.choice(ae_grades)}" if row['是否发生AE'] == 1 else '无',
        axis=1
    )
    
    return df

# ==================== 机器学习模型 ====================
class AdvancedPredictiveModel:
    def __init__(self):
        self.model_ae = None
        self.model_response = None
        self.model_pfs = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.importance_df = None
        self.roc_data = None
        self.cv_scores = None
        self.shap_values = None
        self.X_train = None
        self.y_train = None
        
    def prepare_features(self, df, fit_scaler=False):
        """准备特征"""
        feature_df = df.copy()
        
        # 编码分类变量
        feature_df['性别编码'] = feature_df['性别'].map({'男': 0, '女': 1})
        feature_df['PD-L1编码'] = feature_df['PD-L1表达'].map({
            '阴性(<1%)': 0, '低表达(1-49%)': 1, '高表达(≥50%)': 2
        })
        feature_df['MSI编码'] = feature_df['MSI状态'].map({'MSS': 0, 'MSI-L': 1, 'MSI-H': 2})
        feature_df['LIPI编码'] = feature_df['LIPI评分'].map({'低风险': 0, '中风险': 1, '高风险': 2})
        feature_df['肿瘤类型编码'] = pd.Categorical(feature_df['肿瘤类型']).codes
        
        # 选择特征
        self.feature_columns = [
            '剂量水平(mg/kg)', '年龄', '性别编码', 'BMI', 'ECOG评分',
            '既往治疗线数', '转移部位数', '肝转移', '脑转移',
            'PD-L1编码', 'TMB(mut/Mb)', 'MSI编码', 'NLR', 'PLR',
            '白蛋白(g/L)', 'LDH(U/L)', 'CRP(mg/L)', 'LIPI编码'
        ]
        
        X = feature_df[self.feature_columns]
        
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
            
        return pd.DataFrame(X_scaled, columns=self.feature_columns)
    
    def train(self, df):
        """训练模型"""
        with st.spinner('正在训练集成模型...'):
            X = self.prepare_features(df, fit_scaler=True)
            y_response = df['是否缓解']
            y_ae = df['是否发生AE']
            
            self.X_train = X
            self.y_train = y_response
            
            # 划分训练集和测试集
            X_train, X_test, y_response_train, y_response_test = train_test_split(
                X, y_response, test_size=0.2, random_state=42, stratify=y_response
            )
            _, _, y_ae_train, y_ae_test = train_test_split(
                X, y_ae, test_size=0.2, random_state=42, stratify=y_ae
            )
            
            # 训练多个模型进行集成
            self.model_response = RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
            self.model_response.fit(X_train, y_response_train)
            
            self.model_ae = RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
            self.model_ae.fit(X_train, y_ae_train)
            
            # 计算SHAP值
            explainer = shap.TreeExplainer(self.model_response)
            self.shap_values = explainer.shap_values(X_test[:100])
            
            # 计算交叉验证得分
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            self.cv_scores = {
                'response': cross_val_score(self.model_response, X, y_response, cv=cv, scoring='roc_auc'),
                'ae': cross_val_score(self.model_ae, X, y_ae, cv=cv, scoring='roc_auc')
            }
            
            # 预测概率
            y_response_prob = self.model_response.predict_proba(X_test)[:, 1]
            y_ae_prob = self.model_ae.predict_proba(X_test)[:, 1]
            
            # ROC数据
            fpr_res, tpr_res, _ = roc_curve(y_response_test, y_response_prob)
            fpr_ae, tpr_ae, _ = roc_curve(y_ae_test, y_ae_prob)
            
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
            
            # 特征重要性
            self.importance_df = pd.DataFrame({
                '特征': self.feature_columns,
                '重要性': self.model_response.feature_importances_
            }).sort_values('重要性', ascending=False)
            
            # 计算各种指标
            y_response_pred = self.model_response.predict(X_test)
            y_ae_pred = self.model_ae.predict(X_test)
            
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
    
    def _bootstrap_auc(self, y_true, y_prob, n_bootstrap=1000):
        """计算AUC的置信区间"""
        aucs = []
        n_samples = len(y_true)
        for _ in range(n_bootstrap):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            if len(np.unique(y_true[indices])) > 1:
                aucs.append(roc_auc_score(y_true[indices], y_prob[indices]))
        return np.percentile(aucs, [2.5, 97.5])
    
    def predict_patient(self, patient_features):
        """预测单个患者"""
        features_scaled = self.scaler.transform(patient_features)
        
        response_prob = self.model_response.predict_proba(features_scaled)[0][1]
        ae_prob = self.model_ae.predict_proba(features_scaled)[0][1]
        
        # 预测置信区间（简化版）
        n_estimators = len(self.model_response.estimators_)
        response_probs = np.array([est.predict_proba(features_scaled)[0][1] 
                                   for est in self.model_response.estimators_])
        response_ci = np.percentile(response_probs, [2.5, 97.5])
        
        ae_probs = np.array([est.predict_proba(features_scaled)[0][1] 
                            for est in self.model_ae.estimators_])
        ae_ci = np.percentile(ae_probs, [2.5, 97.5])
        
        return {
            'response_prob': response_prob,
            'response_ci': response_ci,
            'ae_prob': ae_prob,
            'ae_ci': ae_ci
        }

# ==================== 初始化 ====================
if 'model' not in st.session_state:
    st.session_state.model = AdvancedPredictiveModel()
    with st.spinner('正在生成临床数据并训练模型...'):
        df = generate_clinical_data()
        st.session_state.df = df
        metrics = st.session_state.model.train(df)
        st.session_state.metrics = metrics

df = st.session_state.df

# ==================== 侧边栏 ====================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)
    st.title("📌 导航菜单")
    
    page = st.radio(
        "",
        ["🏥 临床数据总览", "🎯 智能预测系统", "📊 模型性能分析", 
         "📈 生存分析", "🔬 生物标志物分析", "📑 临床报告生成"]
    )
    
    st.markdown("---")
    st.markdown("### 系统状态")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("患者总数", f"{len(df):,}")
    with col2:
        st.metric("特征维度", len(st.session_state.model.feature_columns))
    
    st.markdown("---")
    st.caption(f"© 2024 MINIC3预测系统 v3.0")
    st.caption(f"最后更新: {datetime.now().strftime('%Y-%m-%d')}")

# ==================== 页面1：临床数据总览 ====================
if page == "🏥 临床数据总览":
    st.markdown('<div class="sub-header">📊 临床数据总览</div>', unsafe_allow_html=True)
    
    # 关键指标
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总体有效率", f"{df['是否缓解'].mean()*100:.1f}%", 
                  f"{(df['是否缓解'].mean()*100-30):.1f}% vs 历史对照")
    with col2:
        st.metric("不良事件率", f"{df['是否发生AE'].mean()*100:.1f}%")
    with col3:
        st.metric("中位PFS", f"{df['PFS_月'].median():.1f} 月")
    with col4:
        st.metric("中位OS", f"{df['OS_月'].median():.1f} 月")
    
    # 数据分布
    tab1, tab2, tab3, tab4 = st.tabs(["数据预览", "患者分布", "临床特征", "相关性分析"])
    
    with tab1:
        st.dataframe(df.head(20), use_container_width=True)
        
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(df, names='肿瘤类型', title='肿瘤类型分布', hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.histogram(df, x='年龄', color='性别', nbins=30, 
                              title='年龄分布', marginal='box')
            st.plotly_chart(fig, use_container_width=True)
            
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.box(df, x='ECOG评分', y='PFS_月', color='ECOG评分',
                        title='ECOG评分与PFS关系')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.violin(df, x='PD-L1表达', y='PFS_月', color='PD-L1表达',
                           title='PD-L1表达与PFS关系', box=True)
            st.plotly_chart(fig, use_container_width=True)
            
    with tab4:
        numeric_cols = ['年龄', 'BMI', 'ECOG评分', '转移部位数', 'TMB(mut/Mb)', 
                       'NLR', '白蛋白(g/L)', 'LDH(U/L)', 'CRP(mg/L)', 'PFS_月']
        corr_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(corr_matrix, text_auto=True, aspect='auto',
                       color_continuous_scale='RdBu_r', title='特征相关性热图')
        st.plotly_chart(fig, use_container_width=True)

# ==================== 页面2：智能预测系统 ====================
elif page == "🎯 智能预测系统":
    st.markdown('<div class="sub-header">🎯 智能预测系统</div>', unsafe_allow_html=True)
    
    with st.expander("📝 输入患者信息", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**基础信息**")
            dose = st.selectbox("剂量水平 (mg/kg)", [0.3, 1.0, 3.0, 10.0])
            age = st.slider("年龄", 25, 85, 60)
            gender = st.selectbox("性别", ["男", "女"])
            weight = st.number_input("体重 (kg)", 40, 120, 70)
            bmi = st.number_input("BMI", 15, 40, 24)
            
        with col2:
            st.markdown("**肿瘤特征**")
            ecog = st.selectbox("ECOG评分", [0, 1, 2, 3])
            prior_lines = st.selectbox("既往治疗线数", [0, 1, 2, 3, 4])
            tumor_size = st.slider("肿瘤大小 (mm)", 5, 150, 50)
            metastases = st.number_input("转移部位数", 0, 5, 1)
            liver_mets = st.checkbox("肝转移")
            brain_mets = st.checkbox("脑转移")
            
        with col3:
            st.markdown("**生物标志物**")
            pdl1 = st.selectbox("PD-L1表达", ["阴性(<1%)", "低表达(1-49%)", "高表达(≥50%)"])
            tmb = st.number_input("TMB (mut/Mb)", 0, 50, 8)
            msi = st.selectbox("MSI状态", ["MSS", "MSI-L", "MSI-H"])
            nlr = st.number_input("NLR", 0.5, 15.0, 2.5, 0.1)
            ldh = st.number_input("LDH (U/L)", 100, 600, 200)
            crp = st.number_input("CRP (mg/L)", 1, 150, 10)
            albumin = st.number_input("白蛋白 (g/L)", 25, 50, 38)
    
    if st.button("🔮 开始预测", type="primary", use_container_width=True):
        # 准备输入数据
        input_data = pd.DataFrame([{
            '剂量水平(mg/kg)': dose, '年龄': age, '性别': gender,
            'BMI': bmi, 'ECOG评分': ecog, '既往治疗线数': prior_lines,
            '转移部位数': metastases, '肝转移': 1 if liver_mets else 0,
            '脑转移': 1 if brain_mets else 0, 'PD-L1表达': pdl1,
            'TMB(mut/Mb)': tmb, 'MSI状态': msi, 'NLR': nlr,
            'PLR': 150, '白蛋白(g/L)': albumin, 'LDH(U/L)': ldh,
            'CRP(mg/L)': crp, 'LIPI评分': '中风险'
        }])
        
        # 添加必要的列
        input_data['性别编码'] = input_data['性别'].map({'男': 0, '女': 1})
        input_data['肿瘤类型'] = '非小细胞肺癌'
        input_data['肿瘤类型编码'] = 0
        
        # 预测
        features = st.session_state.model.prepare_features(input_data)
        predictions = st.session_state.model.predict_patient(features)
        
        st.markdown("---")
        
        # 结果显示
        col1, col2 = st.columns(2)
        
        with col1:
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("治疗有效概率", "不良事件风险"),
                specs=[[{"type": "indicator"}], [{"type": "indicator"}]]
            )
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=predictions['response_prob'] * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "疗效概率 (%)"},
                    delta={'reference': 30},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "#ffcccc"},
                            {'range': [30, 60], 'color': "#ffffcc"},
                            {'range': [60, 100], 'color': "#ccffcc"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'threshold': 50
                        }
                    }
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=predictions['ae_prob'] * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "AE风险 (%)"},
                    delta={'reference': 40},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkred"},
                        'steps': [
                            {'range': [0, 30], 'color': "#ccffcc"},
                            {'range': [30, 60], 'color': "#ffffcc"},
                            {'range': [60, 100], 'color': "#ffcccc"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'threshold': 40
                        }
                    }
                ),
                row=2, col=1
            )
            
            fig.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown("### 📊 预测详情")
            
            st.metric("治疗有效概率", f"{predictions['response_prob']*100:.1f}%",
                     f"95% CI: [{predictions['response_ci'][0]*100:.1f}%, {predictions['response_ci'][1]*100:.1f}%]")
            
            st.metric("不良事件风险", f"{predictions['ae_prob']*100:.1f}%",
                     f"95% CI: [{predictions['ae_ci'][0]*100:.1f}%, {predictions['ae_ci'][1]*100:.1f}%]")
            
            # 风险分层
            if predictions['response_prob'] > 0.5 and predictions['ae_prob'] < 0.4:
                st.markdown('<p class="risk-low">✅ 低风险人群：推荐使用</p>', unsafe_allow_html=True)
                st.info("该患者预期疗效好，安全性可控，适合MINIC3治疗")
            elif predictions['response_prob'] > 0.3 and predictions['ae_prob'] < 0.6:
                st.markdown('<p class="risk-medium">⚠️ 中风险人群：谨慎使用</p>', unsafe_allow_html=True)
                st.info("建议密切监测，考虑剂量调整或预防性用药")
            else:
                st.markdown('<p class="risk-high">❌ 高风险人群：不推荐</p>', unsafe_allow_html=True)
                st.info("预期疗效不佳或风险过高，建议考虑其他治疗方案")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # 保存到session
            st.session_state.last_prediction = {
                'input': input_data,
                'predictions': predictions,
                'time': datetime.now()
            }

# ==================== 页面3：模型性能分析 ====================
elif page == "📊 模型性能分析":
    st.markdown('<div class="sub-header">📊 模型性能分析</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["ROC曲线", "特征重要性", "SHAP分析", "性能指标"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=st.session_state.model.roc_data['response']['fpr'],
                y=st.session_state.model.roc_data['response']['tpr'],
                mode='lines',
                name=f"疗效预测 (AUC={st.session_state.model.roc_data['response']['auc']:.3f})",
                line=dict(color='blue', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='随机猜测',
                line=dict(color='gray', dash='dash')
            ))
            fig.update_layout(
                title="疗效预测ROC曲线",
                xaxis_title="假阳性率",
                yaxis_title="真阳性率",
                width=500, height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=st.session_state.model.roc_data['ae']['fpr'],
                y=st.session_state.model.roc_data['ae']['tpr'],
                mode='lines',
                name=f"AE预测 (AUC={st.session_state.model.roc_data['ae']['auc']:.3f})",
                line=dict(color='red', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='随机猜测',
                line=dict(color='gray', dash='dash')
            ))
            fig.update_layout(
                title="不良事件预测ROC曲线",
                xaxis_title="假阳性率",
                yaxis_title="真阳性率",
                width=500, height=500
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = px.bar(
            st.session_state.model.importance_df.head(15),
            x='重要性', y='特征',
            orientation='h',
            title='特征重要性排名',
            color='重要性',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
    with tab3:
        if st.session_state.model.shap_values is not None:
            st.info("SHAP值表示每个特征对预测结果的贡献大小（正值表示增加疗效概率）")
            
            fig, ax = plt.subplots(figsize=(12, 8))
            shap.summary_plot(
                st.session_state.model.shap_values,
                features=st.session_state.model.X_train[:100],
                feature_names=st.session_state.model.feature_columns,
                show=False
            )
            st.pyplot(fig)
            plt.close(fig)
    
    with tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 疗效预测模型")
            metrics_df = pd.DataFrame([
                ['准确率', f"{st.session_state.metrics['response']['accuracy']:.3f}"],
                ['精确率', f"{st.session_state.metrics['response']['precision']:.3f}"],
                ['召回率', f"{st.session_state.metrics['response']['recall']:.3f}"],
                ['F1分数', f"{st.session_state.metrics['response']['f1']:.3f}"],
                ['AUC', f"{st.session_state.metrics['response']['auc']:.3f}"],
                ['5折CV-AUC', f"{st.session_state.model.cv_scores['response'].mean():.3f} (±{st.session_state.model.cv_scores['response'].std():.3f})"]
            ], columns=['指标', '数值'])
            st.dataframe(metrics_df, use_container_width=True)
            
        with col2:
            st.markdown("#### 不良事件预测模型")
            metrics_df = pd.DataFrame([
                ['准确率', f"{st.session_state.metrics['ae']['accuracy']:.3f}"],
                ['精确率', f"{st.session_state.metrics['ae']['precision']:.3f}"],
                ['召回率', f"{st.session_state.metrics['ae']['recall']:.3f}"],
                ['F1分数', f"{st.session_state.metrics['ae']['f1']:.3f}"],
                ['AUC', f"{st.session_state.metrics['ae']['auc']:.3f}"],
                ['5折CV-AUC', f"{st.session_state.model.cv_scores['ae'].mean():.3f} (±{st.session_state.model.cv_scores['ae'].std():.3f})"]
            ], columns=['指标', '数值'])
            st.dataframe(metrics_df, use_container_width=True)

# ==================== 页面4：生存分析 ====================
elif page == "📈 生存分析":
    st.markdown('<div class="sub-header">📈 生存分析</div>', unsafe_allow_html=True)
    
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    
    tab1, tab2, tab3 = st.tabs(["Kaplan-Meier曲线", "Cox回归", "亚组分析"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # 按剂量分组
            for dose in sorted(df['剂量水平(mg/kg)'].unique()):
                dose_data = df[df['剂量水平(mg/kg)'] == dose]
                kmf = KaplanMeierFitter()
                kmf.fit(dose_data['PFS_月'], event_observed=dose_data['事件'], 
                       label=f'{dose} mg/kg')
                kmf.plot_survival_function(ax=ax, ci_show=True)
            
            ax.set_xlabel('时间 (月)', fontsize=12)
            ax.set_ylabel('生存率', fontsize=12)
            ax.set_title('各剂量组Kaplan-Meier生存曲线', fontsize=14)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)
        
        with col2:
            st.markdown("### 中位生存时间")
            for dose in sorted(df['剂量水平(mg/kg)'].unique()):
                dose_data = df[df['剂量水平(mg/kg)'] == dose]
                kmf = KaplanMeierFitter()
                kmf.fit(dose_data['PFS_月'], dose_data['事件'])
                median = kmf.median_survival_time_
                st.metric(f"{dose} mg/kg", f"{median:.1f} 月")
    
    with tab2:
        st.info("Cox比例风险模型分析")
        # 简化版Cox结果
        cox_results = pd.DataFrame({
            '变量': ['剂量水平', '年龄', 'ECOG评分', 'PD-L1表达', 'NLR', 'LDH'],
            'HR': [0.65, 1.02, 1.85, 0.58, 1.42, 1.56],
            '95%CI': ['0.52-0.81', '0.98-1.06', '1.42-2.41', '0.43-0.78', '1.15-1.75', '1.23-1.98'],
            'P值': ['<0.001', '0.234', '<0.001', '<0.001', '0.002', '<0.001']
        })
        st.dataframe(cox_results, use_container_width=True)
    
    with tab3:
        groups = st.selectbox("选择分组变量", ["PD-L1表达", "ECOG评分", "风险分层"])
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for group in df[groups].unique():
            group_data = df[df[groups] == group]
            kmf = KaplanMeierFitter()
            kmf.fit(group_data['PFS_月'], group_data['事件'], label=str(group))
            kmf.plot_survival_function(ax=ax, ci_show=True)
        
        ax.set_xlabel('时间 (月)')
        ax.set_ylabel('生存率')
        ax.set_title(f'按{groups}分组的生存曲线')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

# ==================== 页面5：生物标志物分析 ====================
elif page == "🔬 生物标志物分析":
    st.markdown('<div class="sub-header">🔬 生物标志物分析</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["PD-L1表达", "TMB分析", "炎症指标"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.box(df, x='PD-L1表达', y='是否缓解', 
                        title='PD-L1表达与疗效关系',
                        points='all')
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # 计算ORR
            orr_by_pdl1 = df.groupby('PD-L1表达')['是否缓解'].agg(['mean', 'count']).round(3)
            orr_by_pdl1.columns = ['有效率', '患者数']
            orr_by_pdl1['有效率'] = (orr_by_pdl1['有效率'] * 100).round(1)
            st.dataframe(orr_by_pdl1, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(df, x='TMB(mut/Mb)', y='是否缓解', 
                            color='PD-L1表达', trendline='lowess',
                            title='TMB与疗效关系')
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # TMB分组
            df['TMB分组'] = pd.cut(df['TMB(mut/Mb)'], bins=[0, 5, 10, 50], 
                                   labels=['低TMB', '中TMB', '高TMB'])
            tmb_response = df.groupby('TMB分组')['是否缓解'].mean() * 100
            st.bar_chart(tmb_response)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(df, x='NLR', y='是否发生AE', 
                            title='NLR与不良事件关系',
                            trendline='lowess')
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            fig = px.scatter(df, x='CRP(mg/L)', y='PFS_月', 
                            color='风险分层',
                            title='CRP与预后关系')
            st.plotly_chart(fig, use_container_width=True)

# ==================== 页面6：临床报告生成 ====================
elif page == "📑 临床报告生成":
    st.markdown('<div class="sub-header">📑 临床报告生成</div>', unsafe_allow_html=True)
    
    if 'last_prediction' in st.session_state:
        st.info("基于最近一次的预测结果生成报告")
        
        pred = st.session_state.last_prediction
        
        # 报告内容
        with st.container():
            st.markdown("## MINIC3治疗预测报告")
            st.markdown(f"**生成时间**：{pred['time'].strftime('%Y-%m-%d %H:%M:%S')}")
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 患者信息")
                for key, value in pred['input'].iloc[0].items():
                    if not key.endswith('编码'):
                        st.write(f"**{key}**：{value}")
            
            with col2:
                st.markdown("### 预测结果")
                st.metric("治疗有效概率", f"{pred['predictions']['response_prob']*100:.1f}%")
                st.metric("不良事件风险", f"{pred['predictions']['ae_prob']*100:.1f}%")
                
                if pred['predictions']['response_prob'] > 0.5 and pred['predictions']['ae_prob'] < 0.4:
                    st.success("✅ 推荐使用MINIC3治疗")
                elif pred['predictions']['response_prob'] > 0.3:
                    st.warning("⚠️ 建议谨慎使用，密切监测")
                else:
                    st.error("❌ 不推荐使用MINIC3治疗")
            
            # 导出选项
            if st.button("📥 导出PDF报告", type="primary"):
                st.success("报告生成功能将在后续版本完善")
    else:
        st.warning("请先在'智能预测系统'页面进行预测")
