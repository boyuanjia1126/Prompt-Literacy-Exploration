import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# 设置网页标题
st.set_page_config(page_title="Prompt Literacy Predictor", layout="centered")
st.title("📊 Prompt Literacy 预测工具")
st.markdown("使用眼动数据预测被试的 Prompt Literacy 等级（高 / 中 / 低）")

# 上传CSV文件
uploaded_file = st.file_uploader("请上传包含眼动特征的CSV文件：", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ 文件上传成功，预览如下：")
    st.dataframe(df.head())

    # 检查是否包含必要字段
    required_columns = [
        "fixation_duration_input",
        "fixation_duration_output",
        "fixation_count_input",
        "fixation_count_output",
        "revisits_input",
        "average_fixation_duration"
    ]

    if all(col in df.columns for col in required_columns):
        # 加载训练好的模型和标准化器（需要先训练并保存）
        try:
            model = joblib.load("model/prompt_lit_model.pkl")
            scaler = joblib.load("model/scaler.pkl")

            # 特征提取并标准化
            X = df[required_columns]
            X_scaled = scaler.transform(X)

            # 预测
            preds = model.predict(X_scaled)
            pred_df = df.copy()
            pred_df["Predicted Literacy"] = preds

            st.subheader("📈 预测结果：")
            st.dataframe(pred_df[["Predicted Literacy"] + required_columns])

            # 下载结果
            csv = pred_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 下载预测结果CSV", csv, "predicted_prompt_literacy.csv", "text/csv")

        except FileNotFoundError:
            st.error("❌ 未找到模型文件，请先训练模型并保存在 model/ 目录下。")
    else:
        st.warning(f"⚠️ CSV中缺少必要字段。请确保包含以下列：{', '.join(required_columns)}")
else:
    st.info("👈 请在左侧上传一个CSV文件以开始预测。")
