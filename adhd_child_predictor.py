import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# ==========================================
# 1. 页面配置与模型包加载
# ==========================================
st.set_page_config(page_title="ADHD Risk Prediction", layout="centered")

MODEL_FILE = 'ESPM_ADHD_RandomForest_Final.pkl'

@st.cache_resource
def load_model_package():
    return joblib.load(MODEL_FILE)

# 初始化 SHAP 解释器
@st.cache_resource
def get_shap_explainer(_estimator):
    # TreeExplainer 直接初始化即可
    return shap.TreeExplainer(_estimator)

try:
    # 1. 加载整个数据包
    package = load_model_package()
    
    # 2. 解包
    full_pipeline = package['pipeline']
    youden_threshold = package['threshold']
    feature_names = package['features']
    
    # 3. 提取最终分类器 (Random Forest)
    final_estimator = full_pipeline.named_steps['clf']
    
    # 4. 初始化解释器
    shap_explainer = get_shap_explainer(final_estimator)
    
except Exception as e:
    st.error(f"无法加载模型文件，请确认 '{MODEL_FILE}' 在同一目录下。错误信息: {e}")
    st.stop()

# ==========================================
# 2. 选项定义
# ==========================================
perf_options = {1: 'Top few (1)', 2: 'Above average (2)', 3: 'Average (3)', 4: 'Below average (4)'}
rel_options = {1: 'Good (1)', 2: 'Average (2)', 3: 'Poor (3)'}
symptom_freq_options = {1: 'Not at all (1)', 2: 'A little (2)', 3: 'Quite a bit (3)', 4: 'Very much (4)'}
suicide_options = {0: 'None (0)', 1: 'Suicidal thoughts (1)', 2: 'Suicidal behavior (2)'}
incontinence_freq_options = {0: 'None', 1: '<1/week', 2: '1/week', 3: '2-3/week', 4: '4-5/week', 5: 'Almost every time'}
parent_anx_options = {0: 'None (Normal)', 1: 'Mild', 2: 'Moderate', 3: 'Severe'}

# 界面显示名称映射
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
# 3. 用户输入表单
# ==========================================
st.title("ADHD Risk Assessment Tool")
st.markdown("### Early Screening Prediction Model (ESPM-ADHD)")
st.info(f"System Status: Model Loaded | Clinical Threshold: {youden_threshold:.3%}")

with st.form("adhd_form"):
    st.subheader("1. School & Social")
    c1, c2 = st.columns(2)
    with c1:
        child_chinese = st.selectbox(feature_map_display['child_chinese'], options=list(perf_options.keys()), format_func=lambda x: perf_options[x])
        child_relationships = st.selectbox(feature_map_display['child_relationships'], options=list(rel_options.keys()), format_func=lambda x: rel_options[x])
    with c2:
        child_math = st.selectbox(feature_map_display['child_math'], options=list(perf_options.keys()), format_func=lambda x: perf_options[x])

    st.subheader("2. Mental Health")
    c1, c2 = st.columns(2)
    with c1:
        child_depression = st.selectbox(feature_map_display['child_depression'], options=list(symptom_freq_options.keys()), format_func=lambda x: symptom_freq_options[x])
        child_suicide = st.selectbox(feature_map_display['child_suicide'], options=list(suicide_options.keys()), format_func=lambda x: suicide_options[x])
    with c2:
        child_anxiety = st.selectbox(feature_map_display['child_anxiety'], options=list(symptom_freq_options.keys()), format_func=lambda x: symptom_freq_options[x])

    st.subheader("3. Rutter Scores")
    c1, c2 = st.columns(2)
    with c1:
        rutter_score_a = st.number_input(feature_map_display['rutter_score_a'], min_value=0, max_value=30, value=0)
    with c2:
        rutter_score_n = st.number_input(feature_map_display['rutter_score_n'], min_value=0, max_value=30, value=0)

    st.subheader("4. Physiological & Habits")
    c1, c2, c3 = st.columns(3)
    with c1:
        urine_enuresis = st.selectbox(feature_map_display['urine_enuresis'], options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        urine_leakage_frequency = st.selectbox(feature_map_display['urine_leakage_frequency'], options=list(incontinence_freq_options.keys()), format_func=lambda x: incontinence_freq_options[x])
        urine_delayed = st.selectbox(feature_map_display['urine_delayed'], options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    with c2:
        stool_stains = st.selectbox(feature_map_display['stool_stains'], options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        stool_constipation = st.selectbox(feature_map_display['stool_constipation'], options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    with c3:
        cshq_daysleep = st.number_input(feature_map_display['cshq_daysleep'], min_value=0, max_value=50, value=8)

    st.subheader("5. Caregiver")
    parent_anxiety_degree = st.selectbox(feature_map_display['parent_anxiety_degree'], options=list(parent_anx_options.keys()), format_func=lambda x: parent_anx_options[x])

    submit_btn = st.form_submit_button("Run Prediction & Explanation")

# ==========================================
# 4. 预测逻辑与 SHAP 解释 (升级为 Waterfall 图)
# ==========================================
if submit_btn:
    # 1. 构造初始数据
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

    st.divider()
    
    with st.spinner('Analyzing...'):
        try:
            # 2. 特征对齐
            input_data_sorted = input_data[feature_names]
            
            # 3. 分步预处理 (跳过 SMOTE)
            step_imputer = full_pipeline.named_steps['imputer']
            step_scaler = full_pipeline.named_steps['scaler']
            
            data_imputed = step_imputer.transform(input_data_sorted)
            transformed_data = step_scaler.transform(data_imputed)
            
            # 4. 预测概率
            prob = full_pipeline.predict_proba(input_data_sorted)[:, 1][0]
            
            # 5. 结果判定
            risk_percent = prob * 100
            threshold_percent = youden_threshold * 100
            
            if prob >= youden_threshold:
                st.error("### ⚠️ Result: High Risk Detected")
                st.markdown(f"""
                **Predicted Probability:** `{risk_percent:.2f}%`  
                *(Threshold for High Risk: > {threshold_percent:.2f}%)*
                """)
                st.warning("**Recommendation:** Based on the ESPM-ADHD model screening criteria, this child shows signs consistent with ADHD risk. **Clinical referral and further diagnostic evaluation are strongly recommended.**")
            else:
                st.success("### ✅ Result: Low Risk Detected")
                st.markdown(f"""
                **Predicted Probability:** `{risk_percent:.2f}%`  
                *(Threshold for High Risk: > {threshold_percent:.2f}%)*
                """)
                st.info("**Recommendation:** No immediate high-risk indicators detected. Routine monitoring and follow-up are suggested.")
            
            # ------------------------------------------------------------------
            # [核心修复] SHAP 可视化升级：使用 Waterfall Plot (瀑布图)
            # 这种图表不仅不会报错，而且解释性更强，非常适合论文发表
            # ------------------------------------------------------------------
            st.divider()
            st.subheader("Model Interpretation (SHAP Waterfall Plot)")
            st.markdown("The chart below details how each factor drives the risk score from the average (E[f(x)]) to the final prediction (f(x)).")
            
            # 1. 使用 explainer 直接调用，获取 Explanation 对象
            # 这比 .shap_values() 更现代，包含更多元数据
            shap_object = shap_explainer(transformed_data)
            
            # 2. 提取正类 (High Risk / Class 1) 的解释对象
            # 对于二分类，shap_object.values 形状通常是 (1, n_features, 2)
            if len(shap_object.values.shape) == 3:
                # 取出: 第0个样本, 所有特征, 第1个类别(ADHD)
                shap_explanation = shap_object[0, :, 1]
            else:
                # 兼容旧版本或特殊情况
                shap_explanation = shap_object[0]

            # 3. 赋予友好的特征名称 (用于图表显示)
            display_names = [feature_map_display.get(f, f) for f in feature_names]
            shap_explanation.feature_names = display_names
            
            # 4. 绘制 Waterfall Plot
            # 创建画布
            fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
            
            # 绘制 (show=False 允许我们在 Streamlit 中手动渲染)
            shap.plots.waterfall(shap_explanation, show=False, max_display=10)
            
            # 渲染到 Streamlit
            st.pyplot(fig, bbox_inches='tight')
            
            # 清理
            plt.clf()
            plt.close()

            st.caption("Interpretation: The bottom value E[f(x)] is the average risk. Red bars push the risk up; Blue bars pull it down. The top value f(x) is the final predicted risk score.")

        except KeyError as e:
            st.error(f"Feature Mismatch Error: Missing feature {e}. Please ensure input data matches the model.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
            import traceback
            st.code(traceback.format_exc())
