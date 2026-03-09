import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="MINIC3预测系统", page_icon="🧠")
st.title("🧠 MINIC3预测系统")

# 生成模拟数据
np.random.seed(42)
n = 200

data = pd.DataFrame({
    '年龄': np.random.randint(30, 80, n),
    '剂量': np.random.choice([0.3, 1, 3, 10], n),
    'ECOG': np.random.choice([0, 1, 2], n),
    'PDL1': np.random.choice(['阴性', '阳性'], n),
    '肿瘤大小': np.random.randint(10, 100, n),
    '疗效': np.random.choice(['有效', '无效'], n, p=[0.4, 0.6]),
    'AE': np.random.choice(['有', '无'], n, p=[0.5, 0.5])
})

st.write("## 📊 数据概览")
st.dataframe(data.head(10))

# 统计指标
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("总患者数", len(data))
with col2:
    st.metric("有效率", f"{ (data['疗效']=='有效').mean()*100:.1f}%")
with col3:
    st.metric("AE率", f"{ (data['AE']=='有').mean()*100:.1f}%")

# 简单分析
st.write("## 📈 数据分析")
option = st.selectbox("选择分析维度", ["年龄分布", "剂量分布", "ECOG分布"])

if option == "年龄分布":
    st.bar_chart(data['年龄'].value_counts().sort_index())
elif option == "剂量分布":
    st.bar_chart(data['剂量'].value_counts())
else:
    st.bar_chart(data['ECOG'].value_counts())

# 简单预测（基于规则）
st.write("## 🎯 快速评估")
col1, col2 = st.columns(2)
with col1:
    age = st.slider("年龄", 30, 80, 60)
    dose = st.selectbox("剂量", [0.3, 1, 3, 10])
    ecog = st.selectbox("ECOG", [0, 1, 2])
with col2:
    pdl1 = st.selectbox("PD-L1", ["阴性", "阳性"])
    tumor = st.slider("肿瘤大小", 10, 100, 50)

if st.button("评估"):
    # 简单的规则引擎
    score = 0
    if dose >= 3: score += 1
    if pdl1 == "阳性": score += 1
    if ecog <= 1: score += 1
    if tumor < 50: score += 1
    
    prob = score / 4
    
    st.write(f"### 预测结果")
    st.write(f"有效概率: {prob:.0%}")
    
    if prob >= 0.75:
        st.success("✅ 高概率有效")
    elif prob >= 0.5:
        st.warning("⚠️ 中等概率有效")
    else:
        st.error("❌ 低概率有效")
