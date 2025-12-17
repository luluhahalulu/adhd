import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# ==========================================
# 1. 页面配置与模型加载
# ==========================================
st.set_page_config(page_title="ADHD Risk Prediction", layout="centered")

# 请确保您的 pkl 文件名为 'adhd_full_model.pkl' 并且与此脚本在同一目录下
MODEL_FILE = 'adhd_full_model.pkl'

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

try:
    model = load_model()
except Exception as e:
    st.error(f"无法加载模型文件 '{MODEL_FILE}'。请确认文件已上传。错误信息: {e}")
    st.stop()

# ==========================================
# 2. 选项映射字典
# ==========================================
perf_options = {1: 'Top few (1)', 2: 'Above average (2)', 3: 'Average (3)', 4: 'Below average (4)'}
rel_options = {1: 'Good (1)', 2: 'Average (2)', 3: 'Poor (3)'}
symptom_freq_options = {1: 'Not at all (1)', 2: 'A little (2)', 3: 'Quite a bit (3)', 4: 'Very much (4)'}
suicide_options = {0: 'None (0)', 1: 'Suicidal thoughts (1)', 2: 'Suicidal behavior (2)'}
incontinence_freq_options = {0: 'None', 1: '<1/week', 2: '1/week', 3: '2-3/week', 4: '4-5/week', 5: 'Almost every time'}
parent_anx_options = {0: 'None (Normal)', 1: 'Mild', 2: 'Moderate', 3: 'Severe'}
binary_options = {0: 'No', 1: 'Yes'}

# ==========================================
# 3. 显示名称映射 (15个特征)
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
    'stool_stains': 'Encopresis (Stool stains)',                 
    'stool_constipation': 'Functional constipation',
    'cshq_daysleep': 'Daytime sleepiness (CSHQ Score)',
    'rutter_score_a': 'Antisocial behavior score (Rutter A)',
    'rutter_score_n': 'Neurotic behavior score (Rutter N)',
    'parent_anxiety_degree': 'Intensity of anxiety symptoms (Caregiver)'
}

# ==========================================
# 4. 用户输入表单界面
# ==========================================
st.title("ADHD Risk Assessment Tool")
st.markdown("Please enter the information below based on the clinical assessment.")

with st.form("adhd_form"):
    st.subheader("1. School Performance & Social")
    col1, col2 = st.columns(2)
    with col1:
        child_chinese = st.selectbox(feature_map_display['child_chinese'], options=list(perf_options.keys()), format_func=lambda x: perf_options[x])
        child_relationships = st.selectbox(feature_map_display['child_relationships'], options=list(rel_options.keys()), format_func=lambda x: rel_options[x])
    with col2:
        child_math = st.selectbox(feature_map_display['child_math'], options=list(perf_options.keys()), format_func=lambda x: perf_options[x])

    st.subheader("2. Mental Health")
    col1, col2 = st.columns(2)
    with col1:
        child_depression = st.selectbox(feature_map_display['child_depression'], options=list(symptom_freq_options.keys()), format_func=lambda x: symptom_freq_options[x])
        child_suicide = st.selectbox(feature_map_display['child_suicide'], options=list(suicide_options.keys()), format_func=lambda x: suicide_options[x])
    with col2:
        child_anxiety = st.selectbox(feature_map_display['child_anxiety'], options=list(symptom_freq_options.keys()), format_func=lambda x: symptom_freq_options[x])

    st.subheader("3. Rutter Behavior Questionnaire")
    col1, col2 = st.columns(2)
    with col1:
        rutter_score_a = st.number_input(feature_map_display['rutter_score_a'], min_value=0, max_value=30, value=0)
    with col2:
        rutter_score_n = st.number_input(feature_map_display['rutter_score_n'], min_value=0, max_value=30, value=0)

    st.subheader("4. Physiological & Habits")
    col1, col2, col3 = st.columns(3)
    with col1:
        urine_enuresis = st.selectbox(feature_map_display['urine_enuresis'], options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        urine_leakage_frequency = st.selectbox(feature_map_display['urine_leakage_frequency'], options=list(incontinence_freq_options.keys()), format_func=lambda x: incontinence_freq_options[x])
        urine_delayed = st.selectbox(feature_map_display['urine_delayed'], options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    with col2:
        stool_stains = st.selectbox(feature_map_display['stool_stains'], options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        stool_constipation = st.selectbox(feature_map_display['stool_constipation'], options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    with col3:
        cshq_daysleep = st.number_input(feature_map_display['cshq_daysleep'], min_value=0, max_value=50, value=8)

    st.subheader("5. Caregiver Information")
    parent_anxiety_degree = st.selectbox(feature_map_display['parent_anxiety_degree'], options=list(parent_anx_options.keys()), format_func=lambda x: parent_anx_options[x])

    submit_btn = st.form_submit_button("Run Prediction")


