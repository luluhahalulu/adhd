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
    1: 'Top few',
    2: 'Above average',
    3: 'Average',
    4: 'Below average'
}

# B. 人际关系 (1-3)
rel_options = {
    1: 'Good',
    2: 'Average',
    3: 'Poor'
}

# C. 抑郁/焦虑症状频率 (1-4)
symptom_freq_options = {
    1: 'Not at all',
    2: 'A little',
    3: 'Quite a bit',
    4: 'Very much'
}

# D. 自杀倾向 (0-2)
suicide_options = {
    0: 'None',
    1: 'Suicidal thoughts',
    2: 'Suicidal behavior'
}

# E. 尿失禁频率 (根据文档顺序 0-5)
# Frequency of daytime incontinence
incontinence_freq_options = {
    0: 'None',
    1: '<1/week',
    2: '1/week',
    3: '2-3/week',
    4: '4-5/week',
    5: 'Almost every time around urination'
}

# F. 家长焦虑程度 (0-3)
# 文档只列了 Mild/Moderate/Severe，这里补充 0=None 以覆盖正常情况
parent_anx_options = {
    0: 'None (Normal)',
    1: 'Mild',
    2: 'Moderate',
    3: 'Severe'
}

# G. 二元选项 (0/1)
binary_options = {0: 'No', 1: 'Yes'}

# ==========================================
# 3. 特征名称映射 (模型变量名 -> 显示名称)
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
    'urine_delayed': 'Holding maneuvers (Urinary)', # 对应文档的 Holding maneuvers
    'stool_stains': 'Encopresis',                 # 对应文档的 Encopresis
    'stool_constipation': 'Functional constipation',
    'cshq_daysleep': 'Daytime sleepiness (CSHQ Score)',
    'rutter_score_a': 'Antisocial behavior score (Rutter A)',
    'rutter_score_n': 'Neurotic behavior score (Rutter N)',
    'parent_anxiety_degree': 'Intensity of anxiety symptoms (Caregiver)'
}

# 提取模型所需的 15 个特征列名列表 (顺序至关重要)
model_features = list(feature_map_display.keys())

# ==========================================
# 4. 用户输入表单
# ==========================================
st.title("ADHD Risk Assessment Tool")
st.markdown("Please enter the information below based on the clinical assessment.")

with st.form("adhd_form"):
    
    # --- Section 1: Demographic & School Performance ---
    st.subheader("1. School Performance & Social (学校与社交)")
    col1, col2 = st.columns(2)
    with col1:
        child_chinese = st.selectbox(feature_map_display['child_chinese'], options=list(perf_options.keys()), format_func=lambda x: perf_options[x])
        child_relationships = st.selectbox(feature_map_display['child_relationships'], options=list(rel_options.keys()), format_func=lambda x: rel_options[x])
    with col2:
        child_math = st.selectbox(feature_map_display['child_math'], options=list(perf_options.keys()), format_func=lambda x: perf_options[x])

    # --- Section 2: Mental Health (心理健康) ---
    st.subheader("2. Mental Health (心理健康)")
    col1, col2 = st.columns(2)
    with col1:
        child_depression = st.selectbox(feature_map_display['child_depression'], options=list(symptom_freq_options.keys()), format_func=lambda x: symptom_freq_options[x])
        child_suicide = st.selectbox(feature_map_display['child_suicide'], options=list(suicide_options.keys()), format_func=lambda x: suicide_options[x])
    with col2:
        child_anxiety = st.selectbox(feature_map_display['child_anxiety'], options=list(symptom_freq_options.keys()), format_func=lambda x: symptom_freq_options[x])

    # --- Section 3: Rutter Scale Scores (Rutter量表评分) ---
    st.subheader("3. Rutter Behavior Questionnaire")
    col1, col2 = st.columns(2)
    with col1:
        rutter_score_a = st.number_input(feature_map_display['rutter_score_a'], min_value=0, max_value=30, value=0, help="Score for Antisocial behavior")
    with col2:
        rutter_score_n = st.number_input(feature_map_display['rutter_score_n'], min_value=0, max_value=30, value=0, help="Score for Neurotic behavior")

    # --- Section 4: Physiological & Habits (生理与习惯) ---
    st.subheader("4. Physiological & Habits")
    col1, col2, col3 = st.columns(3)
    with col1:
        # Enuresis / Incontinence
        urine_enuresis = st.selectbox(feature_map_display['urine_enuresis'], options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        urine_leakage_frequency = st.selectbox(feature_map_display['urine_leakage_frequency'], options=list(incontinence_freq_options.keys()), format_func=lambda x: incontinence_freq_options[x])
        urine_delayed = st.selectbox(feature_map_display['urine_delayed'], options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No", help="Holding maneuvers")
    
    with col2:
        # Stool / Constipation
        stool_stains = st.selectbox(feature_map_display['stool_stains'], options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        stool_constipation = st.selectbox(feature_map_display['stool_constipation'], options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    
    with col3:
        # Sleep
        cshq_daysleep = st.number_input(feature_map_display['cshq_daysleep'], min_value=0, max_value=50, value=8)

    # --- Section 5: Caregiver Info (家长信息) ---
    st.subheader("5. Caregiver Information")
    parent_anxiety_degree = st.selectbox(feature_map_display['parent_anxiety_degree'], options=list(parent_anx_options.keys()), format_func=lambda x: parent_anx_options[x])

    # 提交按钮
    submit_btn = st.form_submit_button("Run Prediction")

# ==========================================
# 5. 预测与解释逻辑
# ==========================================
if submit_btn:
    # A. 构造 DataFrame (列名必须与训练时完全一致)
    input_data = pd.DataFrame({
        'rutter_score_a': [rutter_score_a],
        'urine_enuresis': [urine_enuresis],
        'child_suicide': [child_suicide],
        'rutter_score_n': [rutter_score_n],
        'child_depression': [child_depression],
        'parent_anxiety_degree': [parent_anxiety_degree],
        'child_chinese': [child_chinese],
        'child_math': [child_math],
        'stool_stains': [stool_stains],
        'cshq_daysleep': [cshq_daysleep],
        'child_relationships': [child_relationships],
        'urine_leakage_frequency': [urine_leakage_frequency],
        'urine_delayed': [urine_delayed],
        'child_anxiety': [child_anxiety],
        'stool_constipation': [stool_constipation]
    })

    # 确保列顺序正确
    input_data = input_data[model_features]

    st.divider()
    
    with st.spinner('Calculating risk score...'):
        try:
            # 1. 预测
            prediction_cls = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]
            risk_score = prediction_proba[1] * 100  # ADHD 概率 %

            # 2. 显示结果
            if risk_score > 50:
                st.error(f"### High Risk Detected")
                st.write(f"**Predicted Probability of ADHD:** {risk_score:.1f}%")
                st.write("Recommendation: Clinical evaluation is strongly recommended.")
            else:
                st.success(f"### Low Risk Detected")
                st.write(f"**Predicted Probability of ADHD:** {risk_score:.1f}%")
                st.write("Recommendation: Routine monitoring.")

            # 3. SHAP 解释图
            st.subheader("Result Interpretation")
            
            # 从 Pipeline 提取步骤
            # 假设结构: [('imputer',...), ('scaler',...), ('model',...)]
            preprocessor = model[:-1]
            rf_model = model[-1]

            # 转换数据
            data_processed = preprocessor.transform(input_data)

            # 计算 SHAP
            explainer = shap.TreeExplainer(rf_model)
            shap_vals = explainer.shap_values(data_processed)

            # 兼容不同 SHAP 版本 (Binary classification returns list of 2 arrays)
            if isinstance(shap_vals, list):
                shap_val_target = shap_vals[1][0]
                base_val = explainer.expected_value[1]
            else:
                shap_val_target = shap_vals[0]
                base_val = explainer.expected_value

            # 准备显示用的 DataFrame (把列名换成英文描述)
            display_df = input_data.rename(columns=feature_map_display)
            
            # 绘图
            st.write("The chart below shows which factors pushed the risk score up (Red) or down (Blue).")
            shap.force_plot(
                base_val,
                shap_val_target,
                display_df.iloc[0],
                matplotlib=True,
                show=False,
                text_rotation=15
            )
            plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=150)
            st.image("shap_force_plot.png")

        except Exception as e:
            st.error(f"Prediction Error: {e}")
            st.warning("Ensure the input features match the model's training data exactly.")
