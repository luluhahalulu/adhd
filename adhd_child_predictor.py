import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ==========================================
# 1. 页面配置与模型包加载 (关键修改)
# ==========================================
st.set_page_config(page_title="ADHD Risk Prediction", layout="centered")

# 修改为你刚刚保存的新文件名
MODEL_FILE = 'ESPM_ADHD_RandomForest_Final.pkl'

@st.cache_resource
def load_model_package():
    return joblib.load(MODEL_FILE)

try:
    # 加载整个数据包
    package = load_model_package()
    
    # 解包：分别获取模型、阈值和特征列表
    model = package['pipeline']
    youden_threshold = package['threshold']
    feature_names = package['features']
    
except Exception as e:
    st.error(f"无法加载模型文件，请确认 '{MODEL_FILE}' 在同一目录下。错误信息: {e}")
    st.stop()

# ==========================================
# 2. 选项定义 (保持不变)
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
# 3. 用户输入表单 (保持不变)
# ==========================================
st.title("ADHD Risk Assessment Tool")
st.markdown("### Early Screening Prediction Model (ESPM-ADHD)")
st.info(f"System Status: Model Loaded | Clinical Threshold: {youden_threshold:.3%}") # 显示阈值，增加专业感

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

    submit_btn = st.form_submit_button("Run Prediction")

# ==========================================
# 4. 预测逻辑 (核心修改)
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
            # 2. 严格按照训练时的特征顺序对齐数据 (直接使用 pickle 里的 list)
            input_data = input_data[feature_names]
            
            # 3. 运行预测
            # 获取属于"Positive Class (ADHD)"的概率
            prob = model.predict_proba(input_data)[:, 1][0]
            
            # 转换为百分比
            risk_percent = prob * 100
            threshold_percent = youden_threshold * 100
            
            # 4. 使用 Youden 阈值进行判定 (不再是 hardcode 50%)
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
                
        except KeyError as e:
            st.error(f"Feature Mismatch Error: Missing feature {e}. Please ensure input data matches the model.")
        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")
