import streamlit as st

from steps.model_inference_step import predict

st.title("Chest Cancer Classification")

image = st.file_uploader(label="Upload chest CT-Scan image.", type=["jpg", "png"])
get_prediction = st.button("Get Prediction")

if get_prediction:
    cols = st.columns((2,1))
    with cols[0]:
        st.write("### Uploaded Image")
        st.image(image)
    with cols[1]:
        st.write("### Prediction")
        prediction = predict(image)
        st.write("# ")
        st.write(prediction)