import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
import matplotlib.pyplot as plt

# Load the model
model = joblib.load('xgboost_adhd.pkl')

# Define feature options

child_chinese_options = {
    1: 'Top few (1)',
    2: 'Above average (2)',
    3: 'Average (3)',
    4: 'Below average (4)'
}

child_math_options = {
    1: 'Top few (1)',
    2: 'Above average (2)',
    3: 'Average (3)',
    4: 'Below average (4)'
}

child_relationships_options = {
    1: 'Good (1)', 
    2: 'Average (2)', 
    3: 'Poor (3)'
}

child_depression_options = {
    1: 'Not at all (1)',
    2: 'A little (2)',
    3: 'Quite a bit (3)',
    4: 'Very much (4)'
}

child_anxiety_options = {
    1: 'Not at all (1)',
    2: 'A little (2)',
    3: 'Quite a bit (3)',
    4: 'Very much (4)'
}

child_suicide_options = {
    0: 'None (0)', 
    1: 'Suicidal thoughts (1)', 
    2: 'Suicidal behavior (2)'
}

urine_enuresis_options = {
    0: 'No (0)', 
    1: 'Yes (1)'
}

stool_stains_options = {
    0: 'No (0)', 
    1: 'Yes (1)'
}

rutter_abnormal_options = {
    1: 'No (0)', 
    2: 'Yes (1)'
}

parent_anxiety_degree_options = {
    0: 'None (0)', 
    1: 'Mild (1)', 
    2: 'Moderate (2)',
    3: 'Severe (3)'
}

# Define feature names
feature_names = [
    'Chinese language performance', 'Math performance', 'Interpersonal relationships at school', 'Depressive symptoms', 
    'Anxiety symptoms', 'Suicidal situations', 'Enuresis', 'Encopresis', 'Conflict', 
    'Daytime sleepiness', 'Antisocial behavior score', 'Abnormal behavior', 'Intensity of anxiety symptoms'
]

# Streamlit user interface
st.title("ADHD in Children Predictor")

# child_chinese: categorical selection
child_chinese = st.selectbox("Your child's Chinese language performance over the past six months ranks within the class:", 
                             options=list(child_chinese_options.keys()), format_func=lambda x: child_chinese_options[x])

# child_math: categorical selection
child_math = st.selectbox("Your child's math performance over the past six months ranks within the class:", 
                          options=list(child_math_options.keys()), format_func=lambda x: child_math_options[x])

# child_relationships: categorical selection
child_relationships = st.selectbox("Your child's interpersonal relationships with classmates at school:", 
                                   options=list(child_relationships_options.keys()), format_func=lambda x: child_relationships_options[x])

# child_depression_options: categorical selection
child_depression = st.selectbox("The frequency of your child experiencing depressive symptoms, such as feeling down or losing interest, in the past 12 months:", 
                                options=list(child_depression_options.keys()), format_func=lambda x: child_depression_options[x])

# child_anxiety: categorical selection
child_anxiety = st.selectbox("The frequency of your child experiencing anxiety or irritability, in the past 12 months:", 
                             options=list(child_anxiety_options.keys()), format_func=lambda x: child_anxiety_options[x])

# child_suicide: categorical selection
child_suicide = st.selectbox("Has your child ever had suicidal behavior or suicidal thoughts in the past?", 
                             options=list(child_suicide_options.keys()), format_func=lambda x: child_suicide_options[x])

# urine_enuresis: categorical selection
urine_enuresis = st.selectbox("Has your child ever experienced bedwetting that persisted for more than 3 months?", 
                              options=list(urine_enuresis_options.keys()), format_func=lambda x: urine_enuresis_options[x])

# stool_stains: categorical selection
stool_stains = st.selectbox("Has your child frequently had stool marks on their underwear (excluding periods of diarrhea), after the age of 4?", 
                            options=list(stool_stains_options.keys()), format_func=lambda x: stool_stains_options[x])

# family_conflict: numerical input
family_conflict = st.number_input("Based on the Excerpt from the Family Environment Scale (FES), what's the scores of Conflict(-18~18)", 
                                  min_value = -18, max_value = 18, value = 0)

# cshq_daysleep: numerical input
cshq_daysleep = st.number_input("Based on the Children's Sleep Habits Questionnaire (CSHQ), what's the scores of Daytime sleepiness(8~24)", 
                                min_value = 8, max_value = 24, value = 8)

# rutter_score_a:  numerical input
rutter_score_a = st.number_input("Based on the parent-reported Rutter Children’s Behavior Questionnaire (RCBQ), what's the scores of Antisocial behavior (A behavior) score(0~10)", 
                                min_value = 0, max_value = 10, value = 0)

# rutter_abnormal: categorical selection
rutter_abnormal = st.selectbox("Based on the parent-reported Rutter Children’s Behavior Questionnaire (RCBQ), has your child been diagnosed with abnormal behavior?", 
                               options=list(rutter_abnormal_options.keys()), format_func=lambda x: rutter_abnormal_options[x])

# parent_anxiety_degree: categorical selection
parent_anxiety_degree = st.selectbox("Based on the Zung Self-Rating Anxiety Scale (SAS) in caregivers, what's the intensity of anxiety symptoms", 
                               options=list(parent_anxiety_degree_options.keys()), format_func=lambda x: parent_anxiety_degree_options[x])

# Process inputs and make predictions
feature_values = [child_chinese, child_math, child_relationships, child_depression, 
                  child_anxiety, child_suicide, urine_enuresis, stool_stains, family_conflict, 
                  cshq_daysleep, rutter_score_a, rutter_abnormal, parent_anxiety_degree]
features = np.array([feature_values])

if st.button("Predict"):
    # Predict class and probabilities
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # Display prediction results
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # Generate advice based on prediction results
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:
        advice = (
            f"According to our model, your child has a high risk of ADHD. "
            f"The model predicts that your child's probability of having ADHD is {probability:.1f}%. "
            "While this is just an estimate, it suggests that your child may be at significant risk. "
            "I recommend that you consult a pediatric psychiatrist as soon as possible for further evaluation and "
            "to ensure your child receives an accurate diagnosis and necessary treatment."
        )
    else:
        advice = (
            f"According to our model, your child has a low risk of ADHD. "
            f"The model predicts that your child's probability of not having ADHD is {probability:.1f}%. "
            "However, paying attention to your child's mental health is still very important. "
            "I recommend regular check-ups to monitor your child's mental health, "
            "and to seek medical advice promptly if your child experiences any symptoms."
        )

    st.write(advice)
    
    # Calculate SHAP values and display force plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))

    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)

    st.image("shap_force_plot.png")
