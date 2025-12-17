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
    'stool_stains': 'Encopresis (Stool stains)',                 
    'stool_constipation': 'Functional constipation',
    'cshq_daysleep': 'Daytime sleepiness (CSHQ Score)',
    'rutter_score_a': 'Antisocial behavior score (Rutter A)',
    'rutter_score_n': 'Neurotic behavior score (Rutter N)',
    'parent_anxiety_degree': 'Intensity of anxiety symptoms (Caregiver)'
}

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
    # 1. 构造初始 DataFrame (顺序暂时不重要，下面会自动修复)
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

    # ============================================================
    # 【核心修复】自动获取模型需要的特征顺序并重排
    # ============================================================
    try:
        model_features = None
        # 尝试从 Pipeline 的最后一步(随机森林)获取特征名
        if hasattr(model, 'steps'):
             # 通常 Pipeline 的最后一步是 estimator (clf)
             if hasattr(model.steps[-1][1], 'feature_names_in_'):
                 model_features = model.steps[-1][1].feature_names_in_
        
        # 如果 Pipeline 没有保留，或者直接是模型对象
        if model_features is None and hasattr(model, 'feature_names_in_'):
            model_features = model.feature_names_in_
            
        # 如果还是找不到，使用硬编码的 LASSO 筛选结果顺序 (最后一道防线)
        if model_features is None:
            model_features = [
                'rutter_score_a', 'urine_enuresis', 'child_suicide', 'rutter_score_n', 
                'child_depression', 'parent_anxiety_degree', 'child_chinese', 'child_math', 
                'stool_stains', 'cshq_daysleep', 'child_relationships', 'urine_leakage_frequency', 
                'urine_delayed', 'child_anxiety', 'stool_constipation'
            ]
        
        # 强制按照模型的顺序重排
        # list(model_features) 确保它是列表格式
        input_data = input_data[list(model_features)]
        
    except Exception as e:
        st.error(f"特征对齐失败，请检查模型文件。详细错误: {e}")
        st.stop()

    st.divider()
    
    with st.spinner('Calculating risk score...'):
        try:
            # 2. 预测
            # Pipeline 自动处理缺失值和标准化
            prediction_cls = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]
            risk_score = prediction_proba[1] * 100  # ADHD 概率 %

            # 3. 显示结果
            if risk_score > 50:
                st.error(f"### High Risk Detected")
                st.write(f"**Predicted Probability of ADHD:** {risk_score:.1f}%")
                st.write("Recommendation: Clinical evaluation is strongly recommended.")
            else:
                st.success(f"### Low Risk Detected")
                st.write(f"**Predicted Probability of ADHD:** {risk_score:.1f}%")
                st.write("Recommendation: Routine monitoring.")

            # 4. SHAP 解释图
            st.subheader("Result Interpretation")
            
            # 从 Pipeline 提取步骤
            # 假设结构: [('imputer',...), ('scaler',...), ('model',...)]
            # 倒数第一项是模型，前面所有的步骤是预处理
            preprocessor = model[:-1]
            rf_model = model[-1]

            # 转换数据 (Raw -> Processed)
            # SHAP 需要吃 "标准化后" 的数据才能和 "标准化后" 训练的模型对齐
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
