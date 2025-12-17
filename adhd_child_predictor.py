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
    st.error(f"无法加载模型文件 '{MODEL_FILE}'，请确认文件是否存在。错误信息: {e}")
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
# 3. 显示名称映射
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

# ==========================================
# 5. 预测与解释逻辑
# ==========================================
if submit_btn:
    # 构造数据
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

    # --- 修复 1: 自动特征对齐 ---
    try:
        model_features = None
        if hasattr(model, 'steps') and hasattr(model.steps[-1][1], 'feature_names_in_'):
             model_features = model.steps[-1][1].feature_names_in_
        elif hasattr(model, 'feature_names_in_'):
            model_features = model.feature_names_in_
            
        if model_features is None: # 兜底列表
            model_features = ['rutter_score_a', 'urine_enuresis', 'child_suicide', 'rutter_score_n', 'child_depression', 'parent_anxiety_degree', 'child_chinese', 'child_math', 'stool_stains', 'cshq_daysleep', 'child_relationships', 'urine_leakage_frequency', 'urine_delayed', 'child_anxiety', 'stool_constipation']
        
        input_data = input_data[list(model_features)]
    except Exception as e:
        st.error(f"特征顺序对齐失败: {e}")
        st.stop()

    st.divider()
    
    with st.spinner('Calculating...'):
        try:
            # 预测
            prediction_proba = model.predict_proba(input_data)[0]
            risk_score = prediction_proba[1] * 100

            if risk_score > 50:
                st.error(f"### High Risk Detected")
                st.write(f"**Predicted Probability of ADHD:** {risk_score:.1f}%")
                st.write("Recommendation: Clinical evaluation is strongly recommended.")
            else:
                st.success(f"### Low Risk Detected")
                st.write(f"**Predicted Probability of ADHD:** {risk_score:.1f}%")
                st.write("Recommendation: Routine monitoring.")

            # --- 修复 2: 极度健壮的 SHAP 处理 ---
            st.subheader("Result Interpretation")
            
            preprocessor = model[:-1]
            rf_model = model[-1]
            data_processed = preprocessor.transform(input_data)
            explainer = shap.TreeExplainer(rf_model)
            shap_vals = explainer.shap_values(data_processed)

            # 步骤 A: 提取目标类的 SHAP 值 (通常是第2个，代表Positive Class)
            if isinstance(shap_vals, list):
                target_shap = shap_vals[1]
                base_val_raw = explainer.expected_value[1]
            else:
                target_shap = shap_vals
                base_val_raw = explainer.expected_value

            # 步骤 B: 【核心修复】强制把数组压平成 1维 (解决 matplotlib error)
            # 无论它是 (1, 15) 还是 (1, 15, 1)，统统变成 (15,)
            target_shap = np.array(target_shap).squeeze()
            if target_shap.ndim > 1: # 如果压平后还是多维(极少见), 强取第一行
                target_shap = target_shap[0]

            # 步骤 C: 确保 base_value 是纯数字 (解决 float error)
            if isinstance(base_val_raw, (np.ndarray, list)):
                base_val = float(np.array(base_val_raw).item(0)) # 安全取第一个元素
            else:
                base_val = float(base_val_raw)

            # 绘图
            display_df = input_data.rename(columns=feature_map_display)
            shap.force_plot(
                base_value=base_val,
                shap_values=target_shap, 
                features=display_df.iloc[0],
                matplotlib=True,
                show=False,
                text_rotation=15
            )
            plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=150)
            st.image("shap_force_plot.png")

        except Exception as e:
            st.error(f"发生错误: {str(e)}")
            st.warning("建议检查: 输入特征是否与模型训练时完全一致。")
