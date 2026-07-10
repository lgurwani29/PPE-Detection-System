import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import cv2
import tempfile
import os

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="AI-Based PPE Detection System",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

/* Background */

.stApp{
    background:#F4F8FB;
}

/* Sidebar */

[data-testid="stSidebar"]{
    background:#0F172A;
}

[data-testid="stSidebar"] *{
    color:white;
}

/* Headings */

h1{
    color:#1E3A8A;
    font-weight:800;
}

h2,h3{
    color:#1E3A8A;
}

/* Buttons */

.stButton > button{

    width:100%;

    background:#2563EB;

    color:white;

    border:none;

    border-radius:10px;

    padding:12px;

    font-weight:bold;

    transition:0.3s;
}

.stButton > button:hover{

    background:#1D4ED8;

}

/* Metric Cards */

[data-testid="metric-container"]{

    background:white;

    border-radius:15px;

    padding:18px;

    box-shadow:0 4px 15px rgba(0,0,0,.08);

    border-left:6px solid #2563EB;

}

/* File uploader */

[data-testid="stFileUploader"]{

    background:white;

    border-radius:15px;

    padding:15px;

}

/* Hide Streamlit footer */
 /* Sidebar Navigation */

[data-testid="stSidebar"] label {

    font-size:18px !important;

    font-weight:600 !important;

}

[data-testid="stSidebar"] .stRadio label {

    font-size:18px !important;

}

/* Sidebar Title */

[data-testid="stSidebar"] h1 {

    font-size:30px !important;

    font-weight:800 !important;

}

footer{
    visibility:hidden;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_model():

    return YOLO("best.pt")

model = load_model()

# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.title("🦺 AI-Based PPE Detection")

    st.markdown("---")

    page = st.radio(

        "",

        [

            "🏠 Home",

            "📖 About Project",

            "🤖 Model Information",

            "📚 How To Use",

            "🎯 Objectives",

            "⚠ Limitations",

            "🚀 Future Scope",

            "👩‍💻 About Developer"

        ]

    )

# ==========================================================
# HOME PAGE
# ==========================================================

if page == "🏠 Home":

    st.title("🦺 AI-Based Personal Protective Equipment Detection System")

    st.markdown(
        """
### Industrial Safety Monitoring using YOLOv8

Upload an image of workers wearing Personal Protective Equipment (PPE).

The application automatically detects:

- 👤 Human
- ⛑ Helmet
- 🦺 Safety Vest
- 🧤 Gloves
- 🥾 Boots
"""
    )

    st.markdown("---")

    left,right = st.columns([2,1])

    with left:

        uploaded_file = st.file_uploader(

            "Upload Image",

            type=["jpg","jpeg","png"]

        )

    with right:

        confidence = st.slider(

            "Confidence Threshold",

            0.10,

            1.00,

            0.40,

            0.05

        )

        detect = st.button(

            "🔍 Analyze PPE Compliance"

        )
    # ==========================================================
    # WAITING SCREEN
    # ==========================================================

    if uploaded_file is None:

        st.info("👆 Upload an image to begin PPE detection.")

    elif not detect:

        image = Image.open(uploaded_file)

        st.subheader("📷 Image Preview")

        st.image(
            image,
            use_container_width=True
        )

        st.info(
            "Click **🔍 Analyze PPE Compliance** to start detection."
        )

    else:

        # ==========================================================
        # LOAD IMAGE
        # ==========================================================

        image = Image.open(uploaded_file).convert("RGB")

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg"
        ) as temp:

            image.save(temp.name)

            temp_path = temp.name

        with st.spinner("Analyzing image..."):

            results = model.predict(

                source=temp_path,

                conf=confidence,

                verbose=False

            )

        annotated = results[0].plot()

        output_path = "annotated_result.jpg"

        cv2.imwrite(

            output_path,

            cv2.cvtColor(
                annotated,
                cv2.COLOR_RGB2BGR
            )

        )

        left,right = st.columns(2)

        with left:

            st.subheader("📷 Original Image")

            st.image(

                image,

                use_container_width=True

            )

        with right:

            st.subheader("🤖 Detection Result")

            st.image(

                annotated,

                use_container_width=True

            )

        with open(output_path,"rb") as file:

            st.download_button(

                "📥 Download Detection Result",

                file,

                file_name="PPE_Detection_Result.jpg",

                mime="image/jpeg"

            )

        # ==========================================================
        # COUNT OBJECTS
        # ==========================================================

        detected = {}

        class_names = model.names

        for box in results[0].boxes:

            cls = int(box.cls[0])

            label = class_names[cls]

            detected[label] = detected.get(label,0)+1

        humans = detected.get("human",0)

        helmets = detected.get("helmet",0)

        vests = detected.get("vest",0)

        gloves = detected.get("gloves",0)

        boots = detected.get("boots",0)
        # ==========================================================
        # DETECTION ANALYTICS DASHBOARD
        # ==========================================================

        st.markdown("---")

        st.subheader("📊 Detection Analytics Dashboard")

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric("👤 Humans", humans)
        c2.metric("⛑ Helmets", helmets)
        c3.metric("🦺 Vests", vests)
        c4.metric("🧤 Gloves", gloves)
        c5.metric("🥾 Boots", boots)

        # ==========================================================
        # DETECTION DETAILS
        # ==========================================================

        st.markdown("---")

        st.subheader("📋 Detection Summary")

        summary = pd.DataFrame({

            "Detected Object":[

                "Human",

                "Helmet",

                "Safety Vest",

                "Gloves",

                "Boots"

            ],

            "Count":[

                humans,

                helmets,

                vests,

                gloves,

                boots

            ]

        })

        st.dataframe(

            summary,

            use_container_width=True,

            hide_index=True

        )

        # ==========================================================
        # PPE COMPLIANCE
        # ==========================================================

        st.markdown("---")

        st.subheader("🦺 PPE Compliance Assessment")

        missing=[]

        if humans==0:

            st.info(

                "No workers detected."

            )

        else:

            if helmets<humans:

                missing.append("⛑ Helmet")

            if vests<humans:

                missing.append("🦺 Safety Vest")

            if gloves<humans:

                missing.append("🧤 Gloves")

            if boots<humans:

                missing.append("🥾 Safety Boots")

            if len(missing)==0:

                st.success(

                    "✅ All detected workers appear to be equipped with the required PPE."

                )

            else:

                st.warning(

                    "⚠ Possible Missing PPE"

                )

                for item in missing:

                    st.write(f"• {item}")

        # ==========================================================
        # SAFETY SCORE
        # ==========================================================

        st.markdown("---")

        st.subheader("📈 Estimated Safety Score")

        if humans>0:

            total_required=humans*4

            total_detected=(

                helmets+

                vests+

                gloves+

                boots

            )

            score=min(

                round(

                    (total_detected/total_required)*100

                ),

                100

            )

            st.progress(score/100)

            st.metric(

                "Overall PPE Score",

                f"{score}%"

            )
            if score>=90:

                st.success(

                    "Excellent PPE compliance detected."

                )

            elif score>=70:

                st.info(

                    "Good compliance with minor PPE deficiencies."

                )

            elif score>=50:

                st.warning(

                    "Moderate compliance detected."

                )

            else:

                st.error(

                    "Poor PPE compliance detected."

                )

        # ==========================================================
        # SAFETY RECOMMENDATION
        # ==========================================================

        st.markdown("---")

        st.subheader("🚨 Safety Recommendation")

        if humans==0:

            st.info(

                "Upload an image containing workers."

            )

        elif len(missing)==0:

            st.success(

                """
All detected workers appear to be wearing the required PPE.

No immediate safety concerns were identified.
"""

            )

        else:

            st.error(

                """
Potential PPE violations detected.

Recommended Actions

• Verify helmet usage

• Check safety vest compliance

• Ensure gloves are worn

• Verify safety boots

• Perform manual inspection if required
"""

            )
# ==========================================================
# ABOUT PROJECT PAGE
# ==========================================================

elif page == "📖 About Project":

    st.title("📖 About Project")

    st.markdown("""
### AI-Based Personal Protective Equipment Detection System

The AI-Based Personal Protective Equipment (PPE) Detection System is a computer vision application developed to improve workplace safety in industrial environments.

Using a custom-trained YOLOv8 model, the system automatically detects workers and identifies whether essential PPE is present in an uploaded image.

The application aims to reduce manual inspection efforts and demonstrate how Artificial Intelligence can assist in occupational safety monitoring.
""")

    st.markdown("---")

    st.subheader("✨ Key Features")

    col1, col2 = st.columns(2)

    with col1:

        st.success("✔ Human Detection")

        st.success("✔ Helmet Detection")

        st.success("✔ Safety Vest Detection")

    with col2:

        st.success("✔ Gloves Detection")

        st.success("✔ Safety Boots Detection")

        st.success("✔ PPE Compliance Analysis")

    st.markdown("---")

    st.subheader("🏭 Applications")

    st.write("""

• Construction Sites

• Manufacturing Industries

• Mining Operations

• Warehouses

• Oil & Gas Industries

• Smart Industrial Surveillance

""")

# ==========================================================
# MODEL INFORMATION
# ==========================================================

elif page == "🤖 Model Information":

    st.title("🤖 Model Information")

    st.markdown("### Detection Model")

    info1, info2 = st.columns(2)

    with info1:

        st.metric("Model", "YOLOv8 Nano")

        st.metric("Framework", "Ultralytics")

        st.metric("Programming", "Python")

    with info2:

        st.metric("Training Images", "4401")

        st.metric("Detected Classes", "5")

        st.metric("Confidence", f"{confidence:.2f}")

    st.markdown("---")

    st.subheader("Detected Classes")

    table = pd.DataFrame({

        "Class":[

            "Human",

            "Helmet",

            "Safety Vest",

            "Gloves",

            "Boots"

        ],

        "Purpose":[

            "Worker Detection",

            "Head Protection",

            "Body Protection",

            "Hand Protection",

            "Foot Protection"

        ]

    })

    st.dataframe(

        table,

        use_container_width=True,

        hide_index=True

    )
# ==========================================================
# HOW TO USE
# ==========================================================

elif page == "📚 How To Use":

    st.title("📚 How To Use")

    st.markdown("""
Follow these simple steps to analyze PPE compliance:

### Step 1
Upload an image containing workers.

### Step 2
Navigate to the **Home** page.

### Step 3
Adjust the confidence threshold if required.

### Step 4
Click **🔍 Analyze PPE Compliance**.

### Step 5
Review the detection dashboard.

### Step 6
Download the annotated image if required.
""")

    st.success("💡 Tip: Images with good lighting and a clear view of workers provide the best results.")

# ==========================================================
# OBJECTIVES
# ==========================================================

elif page == "🎯 Objectives":

    st.title("🎯 Project Objectives")

    st.markdown("""
### Primary Objectives

- Detect workers automatically.
- Detect essential Personal Protective Equipment.
- Improve workplace safety monitoring.
- Reduce manual inspection efforts.
- Demonstrate practical Computer Vision using YOLOv8.

---

### Expected Outcome

The system provides a quick estimate of PPE compliance from uploaded images and demonstrates how AI can assist industrial safety inspection.
""")

# ==========================================================
# LIMITATIONS
# ==========================================================

elif page == "⚠ Limitations":

    st.title("⚠ Limitations")

    st.warning("""
Like every Computer Vision model, this application has certain limitations.
""")

    st.markdown("""
### Detection accuracy may decrease when:

- Workers are partially hidden.
- Lighting conditions are poor.
- Images are blurry.
- Objects are very small.
- Workers overlap each other.
- PPE is heavily occluded.

---

### Important Note

The PPE Compliance score is an estimate based on detected object counts. It does not associate individual PPE items with specific workers.
""")

# ==========================================================
# FUTURE SCOPE
# ==========================================================

elif page == "🚀 Future Scope":

    st.title("🚀 Future Scope")

    st.markdown("""
### Possible Enhancements

- 🎥 Live webcam detection
- 📹 CCTV integration
- 📱 Mobile application
- ☁ Cloud deployment
- 🔔 Real-time alert generation
- 👷 Worker tracking
- 🧠 Individual worker PPE association
- 📊 Safety violation reports
- 📈 Analytics dashboard
- 🌍 Smart industrial monitoring
""")

# ==========================================================
# DEVELOPER
# ==========================================================

elif page == "👩‍💻 About Developer":

    st.title("👩‍💻 About Developer")

    left, right = st.columns([1,2])

    with left:

        st.markdown("# 👩‍💻")

    with right:

        st.markdown("""
## Lavanya Guruwani

**B.Tech Computer Science & Engineering**

Final Year Project

AI-Based Personal Protective Equipment Detection System

---

### Technologies Used

- Python
- YOLOv8
- Streamlit
- OpenCV
- Pandas
- Pillow

---

Thank you for exploring this project.
""")
# ==========================================================
# FINAL FOOTER
# ==========================================================

st.markdown("---")

st.markdown(
    """
<div style="
background:#FFFFFF;
padding:18px;
border-radius:12px;
box-shadow:0 3px 10px rgba(0,0,0,0.08);
text-align:center;
">

<h4 style="color:#1E3A8A;">
🦺 AI-Based Personal Protective Equipment Detection System
</h4>

<p>
Developed using
<b>YOLOv8</b>,
<b>Python</b>,
<b>OpenCV</b>,
<b>Streamlit</b>
and
<b>Pandas</b>.
</p>

<p style="color:gray;">

Version 1.0

</p>

</div>
""",
unsafe_allow_html=True
)

# ==========================================================
# CLEAN TEMPORARY FILES
# ==========================================================

try:

    if "output_path" in locals():

        if os.path.exists(output_path):

            os.remove(output_path)

    if "temp_path" in locals():

        if os.path.exists(temp_path):

            os.remove(temp_path)

except:

    pass