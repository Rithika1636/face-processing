import streamlit as st
import cv2
import numpy as np
from PIL import Image
from deepface import DeepFace
import tempfile
import os

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Face Processing AI",
    page_icon="🧠",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background: linear-gradient(135deg, #f8f9ff, #eef3ff);
}

.title-box {
    background: linear-gradient(135deg, #6c63ff, #8f5cff, #ff6b9d);
    padding: 30px;
    border-radius: 22px;
    color: white;
    text-align: center;
    margin-bottom: 25px;
    box-shadow: 0px 8px 25px rgba(80,70,150,0.25);
}

.title-box h1 {
    font-size: 42px;
    margin-bottom: 8px;
}

.title-box p {
    font-size: 18px;
}

.card {
    background: white;
    padding: 22px;
    border-radius: 18px;
    box-shadow: 0px 5px 20px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.feature-title {
    color: #5b4bdb;
    font-size: 27px;
    font-weight: 700;
}

.result-box {
    background: #f4f1ff;
    padding: 18px;
    border-radius: 15px;
    border-left: 5px solid #6c63ff;
}

.success-box {
    background: #e9fff3;
    padding: 18px;
    border-radius: 15px;
    border-left: 5px solid #20a464;
}

.error-box {
    background: #fff0f0;
    padding: 18px;
    border-radius: 15px;
    border-left: 5px solid #e53935;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="title-box">
    <h1>🧠 Face Processing AI</h1>
    <p>Template Matching • Viola-Jones • DeepFace • FaceNet</p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚙️ Navigation")

option = st.sidebar.radio(
    "Select Feature",
    [
        "🏠 Home",
        "🔍 Template Matching",
        "👤 Viola-Jones",
        "🧠 DeepFace",
        "🔐 FaceNet"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Upload your image and select an AI/computer-vision "
    "technique to process it."
)


# =========================================================
# HOME
# =========================================================

if option == "🏠 Home":

    st.markdown("""
    <div class="card">
        <h2>✨ Welcome to Face Processing AI</h2>
        <p>
        This application combines classical computer vision and
        deep-learning based face analysis techniques in one platform.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="card">
            <div class="feature-title">🔍 Template Matching</div>
            <p>
            Finds a smaller template image inside a larger image
            using OpenCV template matching.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <div class="feature-title">👤 Viola-Jones</div>
            <p>
            Detects human faces using the Haar Cascade
            implementation of the Viola-Jones algorithm.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <div class="feature-title">🧠 DeepFace</div>
            <p>
            Performs deep-learning based facial analysis including
            age, gender, emotion and other available attributes.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <div class="feature-title">🔐 FaceNet</div>
            <p>
            Generates FaceNet facial embeddings and compares
            two face images using cosine similarity.
            </p>
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# TEMPLATE MATCHING
# =========================================================

elif option == "🔍 Template Matching":

    st.markdown("""
    <div class="card">
        <div class="feature-title">🔍 Template Matching</div>
        <p>
        Upload a main image and a smaller template image.
        The system searches for the template inside the main image.
        </p>
    </div>
    """, unsafe_allow_html=True)

    main_file = st.file_uploader(
        "Upload Main Image",
        type=["jpg", "jpeg", "png", "bmp"],
        key="main_template"
    )

    template_file = st.file_uploader(
        "Upload Template Image",
        type=["jpg", "jpeg", "png", "bmp"],
        key="template"
    )

    if main_file and template_file:

        main_img = Image.open(main_file).convert("RGB")
        template_img = Image.open(template_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                main_img,
                caption="Main Image",
                use_container_width=True
            )

        with col2:
            st.image(
                template_img,
                caption="Template Image",
                use_container_width=True
            )

        if st.button("🚀 Run Template Matching", use_container_width=True):

            main_array = np.array(main_img)
            template_array = np.array(template_img)

            main_gray = cv2.cvtColor(
                main_array,
                cv2.COLOR_RGB2GRAY
            )

            template_gray = cv2.cvtColor(
                template_array,
                cv2.COLOR_RGB2GRAY
            )

            h, w = template_gray.shape

            if h > main_gray.shape[0] or w > main_gray.shape[1]:

                st.error(
                    "Template image must be smaller than the main image."
                )

            else:

                result = cv2.matchTemplate(
                    main_gray,
                    template_gray,
                    cv2.TM_CCOEFF_NORMED
                )

                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(
                    result
                )

                top_left = max_loc

                bottom_right = (
                    top_left[0] + w,
                    top_left[1] + h
                )

                output = main_array.copy()

                cv2.rectangle(
                    output,
                    top_left,
                    bottom_right,
                    (255, 0, 0),
                    4
                )

                st.markdown(
                    f"""
                    <div class="success-box">
                    <b>Template Found</b><br>
                    Matching Score: {max_val:.4f}<br>
                    Location: {top_left}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.image(
                    output,
                    caption="Template Matching Result",
                    use_container_width=True
                )


# =========================================================
# VIOLA JONES
# =========================================================

elif option == "👤 Viola-Jones":

    st.markdown("""
    <div class="card">
        <div class="feature-title">👤 Viola-Jones Face Detection</div>
        <p>
        Uses OpenCV Haar Cascade to detect human faces.
        </p>
    </div>
    """, unsafe_allow_html=True)

    face_file = st.file_uploader(
        "Upload Face Image",
        type=["jpg", "jpeg", "png", "bmp"],
        key="viola"
    )

    if face_file:

        image = Image.open(face_file).convert("RGB")

        st.image(
            image,
            caption="Input Image",
            use_container_width=True
        )

        if st.button(
            "🚀 Detect Faces",
            use_container_width=True
        ):

            img_array = np.array(image)

            gray = cv2.cvtColor(
                img_array,
                cv2.COLOR_RGB2GRAY
            )

            cascade_path = os.path.join(
                cv2.data.haarcascades,
                "haarcascade_frontalface_default.xml"
            )

            face_cascade = cv2.CascadeClassifier(
                cascade_path
            )

            if face_cascade.empty():

                st.error(
                    "Haar Cascade could not be loaded."
                )

            else:

                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )

                output = img_array.copy()

                for (x, y, w, h) in faces:

                    cv2.rectangle(
                        output,
                        (x, y),
                        (x + w, y + h),
                        (0, 255, 0),
                        3
                    )

                st.image(
                    output,
                    caption="Viola-Jones Result",
                    use_container_width=True
                )

                st.markdown(
                    f"""
                    <div class="success-box">
                    <b>Faces Detected:</b> {len(faces)}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# =========================================================
# DEEPFACE
# =========================================================

elif option == "🧠 DeepFace":

    st.markdown("""
    <div class="card">
        <div class="feature-title">🧠 DeepFace Analysis</div>
        <p>
        DeepFace performs deep-learning based facial analysis
        on the uploaded image.
        </p>
    </div>
    """, unsafe_allow_html=True)

    deep_file = st.file_uploader(
        "Upload Face Image",
        type=["jpg", "jpeg", "png", "bmp"],
        key="deepface"
    )

    if deep_file:

        image = Image.open(deep_file).convert("RGB")

        st.image(
            image,
            caption="Input Image",
            use_container_width=True
        )

        if st.button(
            "🚀 Run DeepFace Analysis",
            use_container_width=True
        ):

            temp_path = None

            try:

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".jpg"
                ) as temp:

                    image.save(temp.name)
                    temp_path = temp.name

                with st.spinner(
                    "DeepFace is analyzing the image..."
                ):

                    analysis = DeepFace.analyze(
                        img_path=temp_path,
                        actions=[
                            "age",
                            "gender",
                            "emotion"
                        ],
                        enforce_detection=False
                    )

                if isinstance(analysis, list):
                    result = analysis[0]
                else:
                    result = analysis

                st.markdown(
                    '<div class="success-box">',
                    unsafe_allow_html=True
                )

                st.subheader("📊 Analysis Result")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Age",
                        result.get("age", "N/A")
                    )

                with col2:
                    gender = result.get("dominant_gender", "N/A")
                    st.metric(
                        "Gender",
                        gender
                    )

                with col3:
                    emotion = result.get(
                        "dominant_emotion",
                        "N/A"
                    )

                    st.metric(
                        "Emotion",
                        emotion
                    )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True
                )

                with st.expander("🔎 Detailed Emotion Scores"):

                    emotions = result.get(
                        "emotion",
                        {}
                    )

                    for name, score in emotions.items():

                        st.write(
                            f"**{name.capitalize()}**: "
                            f"{float(score):.2f}%"
                        )

                        st.progress(
                            min(float(score) / 100, 1.0)
                        )

            except Exception as e:

                st.error(
                    "DeepFace processing failed."
                )

                st.code(str(e))

            finally:

                if temp_path and os.path.exists(temp_path):

                    os.remove(temp_path)


# =========================================================
# FACENET
# =========================================================

elif option == "🔐 FaceNet":

    st.markdown("""
    <div class="card">
        <div class="feature-title">🔐 FaceNet Face Verification</div>
        <p>
        FaceNet converts faces into numerical embeddings and
        compares two images using cosine similarity.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:

        file1 = st.file_uploader(
            "Upload First Face",
            type=["jpg", "jpeg", "png", "bmp"],
            key="facenet1"
        )

    with col2:

        file2 = st.file_uploader(
            "Upload Second Face",
            type=["jpg", "jpeg", "png", "bmp"],
            key="facenet2"
        )

    if file1 and file2:

        col1, col2 = st.columns(2)

        image1 = Image.open(file1).convert("RGB")
        image2 = Image.open(file2).convert("RGB")

        with col1:

            st.image(
                image1,
                caption="Face 1",
                use_container_width=True
            )

        with col2:

            st.image(
                image2,
                caption="Face 2",
                use_container_width=True
            )

        if st.button(
            "🚀 Compare Faces",
            use_container_width=True
        ):

            path1 = None
            path2 = None

            try:

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".jpg"
                ) as temp1:

                    image1.save(temp1.name)
                    path1 = temp1.name

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".jpg"
                ) as temp2:

                    image2.save(temp2.name)
                    path2 = temp2.name

                with st.spinner(
                    "Generating FaceNet embeddings..."
                ):

                    embedding1 = DeepFace.represent(
                        img_path=path1,
                        model_name="Facenet",
                        enforce_detection=False
                    )

                    embedding2 = DeepFace.represent(
                        img_path=path2,
                        model_name="Facenet",
                        enforce_detection=False
                    )

                vector1 = np.array(
                    embedding1[0]["embedding"],
                    dtype=np.float32
                )

                vector2 = np.array(
                    embedding2[0]["embedding"],
                    dtype=np.float32
                )

                cosine_similarity = np.dot(
                    vector1,
                    vector2
                ) / (
                    np.linalg.norm(vector1)
                    *
                    np.linalg.norm(vector2)
                )

                similarity_percentage = (
                    float(cosine_similarity) * 100
                )

                st.markdown(
                    f"""
                    <div class="result-box">
                    <h2>🔐 FaceNet Comparison</h2>
                    <h3>Cosine Similarity: {cosine_similarity:.4f}</h3>
                    <p>
                    Similarity Score: 
                    {similarity_percentage:.2f}%
                    </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.progress(
                    max(
                        0.0,
                        min(
                            float((cosine_similarity + 1) / 2),
                            1.0
                        )
                    )
                )

                st.info(
                    "A higher cosine similarity means the generated "
                    "FaceNet embeddings are more similar. This score "
                    "should be interpreted together with the image "
                    "quality and application-specific threshold."
                )

            except Exception as e:

                st.error(
                    "FaceNet processing failed."
                )

                st.code(str(e))

            finally:

                if path1 and os.path.exists(path1):
                    os.remove(path1)

                if path2 and os.path.exists(path2):
                    os.remove(path2)


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <center>
    <b>🧠 Face Processing AI</b><br>
    Computer Vision + Deep Learning
    </center>
    """,
    unsafe_allow_html=True
)