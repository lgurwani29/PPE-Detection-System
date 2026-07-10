import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import pandas as pd
import cv2
import os

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="PPE Detection System",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

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

/* Button */

.stButton>button{

    width:100%;

    background:#2563EB;

    color:white;

    border-radius:10px;

    border:none;

    padding:0.8rem;

    font-size:17px;

    font-weight:bold;
}

.stButton>button:hover{

    background:#1D4ED8;

    color:white;

}

/* Metric Cards */

[data-testid="metric-container"]{

    background:white;

    border-radius:15px;

    padding:18px;

    box-shadow:0 4px 18px rgba(0,0,0,.08);

    border-left:6px solid #2563EB;

}

/* Upload Box */

[data-testid="stFileUploader"]{

    background:white;

    border-radius:15px;

    padding:15px;

}

/* Footer */

footer{

    visibility:hidden;

}

</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():

    return YOLO("best.pt")

model = load_model()

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.title("🦺 PPE Dashboard")

    st.markdown("---")

    st.subheader("Project")

    st.subheader("Project")

    st.write("""
    AI-Based Personal Protective Equipment Detection System

    B.Tech Final Year Project
    """)

    st.markdown("---")

    st.subheader("Developer")

    st.write("""
           
  **Lavanya Guruwani**

  B.Tech Computer Science & Engineering
  """)

    st.markdown("---")

    st.subheader("Detected Classes")

    st.write("""
👤 Human

⛑ Helmet

🦺 Safety Vest

🧤 Gloves

🥾 Boots
""")

    st.markdown("---")

    st.subheader("Technology")

    st.write("""
• Python

• YOLOv8

• Streamlit

• OpenCV

• Pillow
""")

    st.markdown("---")

    st.subheader("Model")

    st.success("YOLOv8 Nano (Custom Trained)")

    st.markdown("---")

    st.info(
"""
Results may vary in crowded scenes,
low lighting,
or partially visible workers.
"""
    )

# =====================================================
# HEADER
# =====================================================

st.title("🦺 AI-Based Personal Protective Equipment Detection System")

st.caption(
    "Real-Time Industrial Safety Monitoring using Computer Vision"
)

st.markdown("---")

left,right=st.columns([2,1])

with left:

    st.write(
"""
Upload an image of workers wearing Personal Protective Equipment.

The model detects:

- Human
- Helmet
- Safety Vest
- Gloves
- Boots
"""
    )

with right:

    confidence=st.slider(

        "Confidence",

        0.10,

        1.00,

        0.40,

        0.05

    )

uploaded_file=st.file_uploader(

    "Upload PPE Image",

    type=["jpg","jpeg","png"]

)

detect=st.button("🚀 Detect PPE")
# =====================================================
# DETECTION
# =====================================================

if uploaded_file is not None and detect:

    image = Image.open(uploaded_file).convert("RGB")

    left_col, right_col = st.columns(2)

    with left_col:

        st.subheader("📷 Original Image")

        st.image(
            image,
            use_container_width=True
        )

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    ) as temp_file:

        image.save(temp_file.name)

        results = model.predict(
            source=temp_file.name,
            conf=confidence,
            verbose=False
        )

    annotated_image = results[0].plot()

    output_path = "annotated_result.jpg"

    cv2.imwrite(
        output_path,
        cv2.cvtColor(
            annotated_image,
            cv2.COLOR_RGB2BGR
        )
    )

    with right_col:

        st.subheader("🤖 AI Detection")

        st.image(
            annotated_image,
            use_container_width=True
        )

    st.download_button(

        label="📥 Download Annotated Image",

        data=open(output_path, "rb"),

        file_name="PPE_Detection_Result.jpg",

        mime="image/jpeg"

    )

    # =================================================
    # COUNT DETECTIONS
    # =================================================

    detected = {}

    class_names = model.names

    for box in results[0].boxes:

        class_id = int(box.cls[0])

        class_name = class_names[class_id]

        detected[class_name] = detected.get(
            class_name,
            0
        ) + 1

    humans = detected.get("human", 0)

    helmets = detected.get("helmet", 0)

    gloves = detected.get("gloves", 0)

    boots = detected.get("boots", 0)

    vests = detected.get("vest", 0)

    st.markdown("---")

    st.subheader("📊 Detection Dashboard")

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:

        st.metric(
            "👤 Humans",
            humans
        )

    with c2:

        st.metric(
            "⛑ Helmets",
            helmets
        )

    with c3:

        st.metric(
            "🦺 Vests",
            vests
        )

    with c4:

        st.metric(
            "🧤 Gloves",
            gloves
        )

    with c5:

        st.metric(
            "🥾 Boots",
            boots
        )
    # =====================================================
    # DETECTION SUMMARY
    # =====================================================

    st.markdown("---")

    st.subheader("📋 Detection Summary")

    summary_df = pd.DataFrame({

        "Object":[
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
        summary_df,
        use_container_width=True,
        hide_index=True
    )

    # =====================================================
    # PPE COMPLIANCE
    # =====================================================

    st.markdown("---")

    st.subheader("🦺 PPE Compliance Assessment")

    missing_items=[]

    if humans==0:

        st.info(
            "No workers were detected in the uploaded image."
        )

    else:

        if helmets < humans:
            missing_items.append("Helmet")

        if vests < humans:
            missing_items.append("Safety Vest")

        if gloves < humans:
            missing_items.append("Gloves")

        if boots < humans:
            missing_items.append("Safety Boots")

        if len(missing_items)==0:

            st.success(
                "✅ All detected workers appear to be wearing the required PPE."
            )

        else:

            st.warning(
                "⚠ Possible Missing PPE"
            )

            for item in missing_items:

                st.write(f"• {item}")

    # =====================================================
    # SAFETY SCORE
    # =====================================================

    st.markdown("---")

    st.subheader("📈 Estimated Safety Score")

    if humans>0:

        required_items=humans*4

        detected_items=(
            helmets+
            vests+
            gloves+
            boots
        )

        safety_score=min(
            round(
                (detected_items/required_items)*100
            ),
            100
        )

        st.progress(
            safety_score/100
        )

        st.metric(
            "Overall PPE Score",
            f"{safety_score}%"
        )

        if safety_score>=90:

            st.success(
                "Excellent PPE compliance."
            )

        elif safety_score>=70:

            st.info(
                "Good compliance with minor deficiencies."
            )

        elif safety_score>=50:

            st.warning(
                "Moderate compliance. Some PPE appears to be missing."
            )

        else:

            st.error(
                "Low compliance. Multiple PPE items appear to be missing."
            )

    # =====================================================
    # SAFETY RECOMMENDATION
    # =====================================================

    st.markdown("---")

    st.subheader("📌 Safety Recommendation")

    if humans==0:

        st.info(
            "Upload an image containing workers to begin analysis."
        )

    elif len(missing_items)==0:

        st.success(
            """
Workers appear to be complying with PPE requirements.

Continue routine safety inspections and maintain PPE standards.
"""
        )

    else:

        st.error(
            """
Potential PPE deficiencies were identified.

Recommendation:

• Verify helmet usage.

• Ensure safety vests are worn.

• Inspect gloves and safety boots.

• Perform manual inspection before permitting work.
"""
        )
    # =====================================================
    # PROJECT INFORMATION
    # =====================================================

    st.markdown("---")

    with st.expander("📘 About This Project", expanded=False):

        st.markdown("""
### Personal Protective Equipment Detection System

This project has been developed to automate the detection of Personal Protective Equipment (PPE) using Computer Vision and Deep Learning.

The application uses a custom-trained **YOLOv8 Nano** model to identify:

- 👤 Human
- ⛑ Helmet
- 🦺 Safety Vest
- 🧤 Gloves
- 🥾 Boots

The primary objective is to assist industries in improving workplace safety by providing automated PPE monitoring.
""")

    # =====================================================
    # MODEL INFORMATION
    # =====================================================

    with st.expander("🤖 Model Information", expanded=False):

        st.markdown(f"""
### Detection Model

**Architecture**

YOLOv8 Nano

**Framework**

Ultralytics

**Input Size**

640 × 640 pixels

**Confidence Threshold**

{confidence:.2f}

**Detected Classes**

- Human
- Helmet
- Safety Vest
- Gloves
- Boots
""")

    # =====================================================
    # HOW TO USE
    # =====================================================

    with st.expander("📖 How to Use", expanded=False):

        st.markdown("""
1. Upload an image.

2. Adjust the confidence threshold if required.

3. Click **Detect PPE**.

4. Review the detection results.

5. Download the annotated image.

6. Analyse the PPE Compliance and Safety Score.
""")

    # =====================================================
    # PROJECT OBJECTIVES
    # =====================================================

    with st.expander("🎯 Project Objectives", expanded=False):

        st.markdown("""
- Detect workers in industrial environments.

- Detect Personal Protective Equipment.

- Estimate PPE compliance.

- Improve workplace safety monitoring.

- Demonstrate real-time object detection using YOLOv8.
""")

    # =====================================================
    # LIMITATIONS
    # =====================================================

    with st.expander("⚠ Limitations", expanded=False):

        st.warning("""
The system performs best on clear images.

Detection accuracy may decrease in:

- Crowded scenes

- Low lighting

- Motion blur

- Occluded workers

- Very small objects
""")

    # =====================================================
    # FUTURE SCOPE
    # =====================================================

    with st.expander("🚀 Future Scope", expanded=False):

        st.markdown("""
Possible future improvements include:

- Live webcam monitoring

- CCTV integration

- Worker tracking

- Real-time alerts

- Attendance integration

- Safety violation logging

- Cloud deployment

- Email/SMS notification system
""")
    # =====================================================
    # DETECTION STATISTICS
    # =====================================================

    st.markdown("---")

    st.subheader("📈 Detection Statistics")

    total_objects = humans + helmets + vests + gloves + boots

    stat1, stat2, stat3 = st.columns(3)

    with stat1:
        st.info(f"**Total Objects Detected**\n\n{total_objects}")

    with stat2:
        if humans > 0:
            st.success(f"**Workers Detected**\n\n{humans}")
        else:
            st.warning("**Workers Detected**\n\n0")

    with stat3:
        detected_classes = sum([
            humans > 0,
            helmets > 0,
            vests > 0,
            gloves > 0,
            boots > 0
        ])

        st.info(f"**Classes Detected**\n\n{detected_classes}/5")

    # =====================================================
    # SYSTEM STATUS
    # =====================================================

    st.markdown("---")

    st.subheader("🖥️ System Status")

    status_col1, status_col2 = st.columns(2)

    with status_col1:

        st.success("🟢 Model Loaded Successfully")

        st.write("YOLOv8 Nano is active and ready for inference.")

    with status_col2:

        st.success("🟢 Detection Completed")

        st.write("Image processed successfully.")

    # =====================================================
    # PERFORMANCE SUMMARY
    # =====================================================

    st.markdown("---")

    st.subheader("⚙️ Performance Summary")

    performance_df = pd.DataFrame({
        "Parameter": [
            "Model",
            "Framework",
            "Input Size",
            "Confidence Threshold",
            "Detected Classes"
        ],
        "Value": [
            "YOLOv8 Nano",
            "Ultralytics",
            "640 × 640",
            f"{confidence:.2f}",
            "5"
        ]
    })

    st.table(performance_df)

    # =====================================================
    # FOOTER
    # =====================================================

    st.markdown("---")

    st.markdown(
        """
<div style='text-align:center;
padding:15px;
background:white;
border-radius:12px;
box-shadow:0 4px 10px rgba(0,0,0,0.08);'>

<h4 style='color:#1E3A8A; margin-bottom:8px;'>
🦺 Personal Protective Equipment Detection System
</h4>

<p>
Developed using <b>YOLOv8</b>, <b>Streamlit</b>, <b>OpenCV</b> and <b>Python</b>
</p>

<p style='color:gray;'>
B.Tech Computer Science Engineering Project
</p>

</div>
""",
        unsafe_allow_html=True
    )
    # =====================================================
    # ICON LEGEND
    # =====================================================

    st.markdown("---")

    with st.expander("ℹ️ Detection Legend", expanded=False):

        legend = pd.DataFrame({

            "Icon":[
                "👤",
                "⛑",
                "🦺",
                "🧤",
                "🥾"
            ],

            "Meaning":[
                "Human",
                "Helmet",
                "Safety Vest",
                "Gloves",
                "Boots"
            ]

        })

        st.dataframe(
            legend,
            use_container_width=True,
            hide_index=True
        )

    # =====================================================
    # DISCLAIMER
    # =====================================================

    st.markdown("---")

    st.info(
        """
This application is intended for educational and research purposes.

The compliance analysis is based on detected object counts and
should not be considered a certified workplace safety assessment.

Actual industrial deployment would require additional validation,
worker-level PPE association, CCTV integration and real-time tracking.
"""
    )

    # =====================================================
    # CLEAN TEMP FILE
    # =====================================================

    try:

        if os.path.exists(output_path):
            os.remove(output_path)

        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)

    except:
        pass

else:

    st.markdown("---")

    st.info(
        """
### 👋 Welcome

Upload an image and press **🚀 Detect PPE**
to begin the analysis.

The system will automatically:

- Detect workers
- Detect helmets
- Detect gloves
- Detect safety boots
- Detect safety vests
- Estimate PPE compliance
- Generate a safety score
- Allow downloading the annotated result
"""
    )

    st.image(
        "res.jpg",
        use_container_width=True
    )

# =====================================================
# FINAL FOOTER
# =====================================================

st.markdown("---")

st.markdown(
"""
<div style="text-align:center;
padding:25px;
font-size:15px;
color:gray;">

<b>Personal Protective Equipment Detection System</b><br><br>

Developed using Python, Streamlit, OpenCV and YOLOv8.<br><br>

© 2026 PPE Detection Project

</div>
""",
unsafe_allow_html=True
)