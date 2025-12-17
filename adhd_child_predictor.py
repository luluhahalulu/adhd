import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# ==========================================
# 1. 页面配置与模型加载
# ==========================================
st.set_page_config(page_title="ADHD Prediction System", layout="centered")

# 请确保您的 pkl 文件名为 'adhd_full_model.pkl' 并且与此脚本在同一目录下
MODEL_FILE = 'adhd_full_model.pkl'

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}. Please ensure '{MODEL_FILE}' is uploaded.")
    st.stop()

# ==========================================
# 2. 选项映射字典 (严格基于您提供的文档)
# ==========================================

# A. 学业与表现 (1-4)
perf_options = {
    1: 'Top few (1)',
    2: 'Above average (2)',
    3: 'Average (3)',
    4: 'Below average (4)'
}

# B. 人际关系 (1-3)
rel_options = {
    1: 'Good (1)',
    2: 'Average (2)',
    3: 'Poor (3)'
}

# C. 抑郁/焦虑症状频率 (1-4)
symptom_freq_options = {
    1: 'Not at all (1)',
    2: 'A little (2)',
    3: 'Quite a bit (3)',
    4: 'Very much (4)'
}

# D. 自杀倾向 (0-2)
suicide_options = {
    0: 'None (0)',
    1: 'Suicidal thoughts (1)',
    2: 'Suicidal behavior (2)'
}

# E. 尿失禁频率 (根据文档顺序 0-5)
incontinence_freq_options = {
    0: 'None',
    1: '<1/week',
    2: '1/week',
    3: '2-3/week',
    4: '4-5/week',
    5: 'Almost every time'
}

# F. 家长焦虑程度 (0-3)
parent_anx_options = {
    0: 'None (Normal)',
    1: 'Mild',
    2: 'Moderate',
    3: 'Severe'
}

# G. 二元选项 (0/1)
binary_options = {0: 'No', 1: 'Yes'}

# ==========================================
# 3. 显示名称映射 (用于界面显示)
# ==========================================
feature_map_display = {
    'child_chinese': 'Chinese language performance',
    'child_math': 'Math performance',
    'child_relationships': 'Interpersonal relationships at school',
    'child_depression': 'Depressive symptoms (Child)',
    'child_anxiety': 'Anxiety symptoms (Child)',
    'child_suicide': 'Suicidal situations',
    'urine_enuresis': 'Enuresis (Bedwetting)',
    'urine_leakage_frequency': 'Frequency of daytime incontinence',
    'urine_delayed': 'Holding maneuvers (Urinary)', 
    'stool_stains': 'Encopresis (
