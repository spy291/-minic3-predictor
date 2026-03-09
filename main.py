import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="MINIC3预测系统", page_icon="🧠")
st.title("🧠 MINIC3预测系统")

# 生成简单数据
np.random.seed(42)
n = 100
data = pd.DataFrame({
    '年龄': np.random.randint(30, 80, n),
    '剂量': np.random.choice([0.3, 1, 3, 10], n),
    'ECOG': np.random.choice([0, 1, 2], n),
    '疗效': np.random.choice([0, 1], n)
})

st.write("数据预览：")
st.dataframe(data.head())

# 训练简单模型
X = data[['年龄', '剂量', 'ECOG']]
y = data['疗效']
model = RandomForestClassifier()
model.fit(X, y)

st.success("✅ 模型训练成功！")

# 简单预测
st.subheader("快速预测")
age = st.slider("年龄", 30, 80, 60)
dose = st.selectbox("剂量", [0.3, 1, 3, 10])
ecog = st.selectbox("ECOG", [0, 1, 2])

if st.button("预测"):
    pred = model.predict([[age, dose, ecog]])[0]
    prob = model.predict_proba([[age, dose, ecog]])[0][1]
    st.write(f"预测结果：{'有效' if pred==1 else '无效'}")
    st.write(f"有效概率：{prob:.2%}")
