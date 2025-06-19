import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Convert and embed image
encoded_bg = get_base64_image("background.jpg")
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('background.jpg');
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
        background-position: center;
    }
    .custom-section {
        background-color: rgba(255, 255, 255, 0.88);
        padding: 2rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    .center-title {
        text-align: center;
        font-size: 2.5rem;
        color: #1e5631;
        font-weight: bold;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

