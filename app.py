import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Convert and embed image
encoded_bg = get_base64_image("background.jpg")
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded_bg}");
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
        background-position: center;
    }}
    .custom-section {{
        background-color: rgba(255, 255, 255, 0.9);
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 0 10px rgba(0,0,0,0.2);
        margin-top: 1rem;
    }}
    .center-title {{
        text-align: center;
        font-size: 2.8rem;
        color: #1e5631;
        font-weight: bold;
        margin-top: 2rem;
    }}
    </style>
    """,
    unsafe_allow_html=True
)
