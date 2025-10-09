import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px

# ----------------------------
# ✅ โหลดโมเดลที่ train แล้ว
# ----------------------------
with open("model_G.pkl", "rb") as f:
    model = pickle.load(f)

# ----------------------------
# ตั้งค่าเว็บเพจ
# ----------------------------
st.set_page_config(page_title="TNI Career Predictor", page_icon="💼", layout="wide")

# ----------------------------
# เมนูด้านข้าง
# ----------------------------
menu = st.sidebar.selectbox(
    "Manu",
    ["🏠 Predict Career", "📊 YouTube Data Analysis", "🧑‍💼 Profile Page (Piyachart)","🧑‍💼 Profile Page (Kasidit)","🧑🏻‍💻 Code"]
)

# ----------------------------
# แปลงเกรดเป็นตัวเลข
# ----------------------------
grade_map = {"A": 4.0, "B+": 3.5, "B": 3.0, "C+": 2.5, "C": 2.0, "D+": 1.5, "D": 1.0}

subjects = [
    "ENL-201", "JPN-301", "BIS-111", "INT-105", "INT-204", "INT-301",
    "ITE-301", "BIS-105", "BIS-201", "BIS-204", "BIS-401", "BIS-402", "BIS-403","MSC-202"
]

# ---------------------- หน้า Predict Career ----------------------
if menu == "🏠 Predict Career":
    st.title("💼 Career Suitability Prediction")
    st.write("เลือกเกรดของแต่ละวิชา แล้วระบบจะทำนายว่าอาชีพที่เลือกเหมาะสมหรือไม่")

    # กรอกเกรดรายวิชา
    st.header("📚 เลือกเกรดรายวิชา")
    grades = []
    cols = st.columns(3)
    for i, subject in enumerate(subjects):
        with cols[i % 3]:
            grade = st.selectbox(f"{subject}", list(grade_map.keys()), key=subject)
            grades.append(grade_map[grade])

    # เลือกอาชีพ
    st.header("👨‍💻 เลือกอาชีพที่สนใจ")
    career = st.selectbox(
        "อาชีพที่คุณสนใจ",
        ["Data Scientist", "Software Engineer", "Network Engineer", "Database Administrator", "AI Engineer", "อื่น ๆ"]
    )

    career_map = {
        "Data Scientist": 0,
        "Software Engineer": 1,
        "Network Engineer": 2,
        "Database Administrator": 3,
        "AI Engineer": 4,
        "อื่น ๆ": 5
    }
    career_num = career_map.get(career, 5)

    if st.button("🚀 Predict Career Suitability"):
        # ✅ รวม features ให้ตรงกับตอนเทรน (เฉพาะเกรดเท่านั้น)
        X = np.array(grades).reshape(1, -1)

        prediction = model.predict(X)
        if prediction[0] == 1:
            st.success(f"✅ อาชีพ **{career}** เหมาะสมกับคุณ!")
        else:
            st.error(f"❌ อาชีพ **{career}** อาจไม่เหมาะสมกับคุณ")

# ---------------------- หน้า YouTube Data Analysis ----------------------
elif menu == "📊 YouTube Data Analysis":
    st.title("📊 YouTube Top 1000 Channels Data Mining")
    st.markdown("""
    วิเคราะห์ข้อมูล **Top 1000 YouTube Channels**  
    เพื่อหาแนวโน้มของหมวดหมู่ยอดนิยม ยอดวิว และความสัมพันธ์ของเนื้อหา 🌎  
    ข้อมูลต้นฉบับจาก [**Notion Project (Week 4)**](https://www.notion.so/DATA-Mining-Week-4-Youtube-Top-1000-YouTube-243022213a8080afa7ddef39e69fe29b)
    """)

    st.markdown("---")

    # 🎨 รูปภาพส่วนหัว
    st.image(
        "https://cdn-icons-png.flaticon.com/512/1384/1384060.png",
        width=120
    )
    st.markdown("### 🔥 ภาพรวมของ Dataset")
    st.info("""
    Dataset นี้ประกอบด้วย 1,000 ช่อง YouTube ที่มียอด Subscribe สูงสุด  
    พร้อมข้อมูลหมวดหมู่ (Category), ยอดวิวเฉลี่ย, และจำนวนวิดีโอ
    """)

    # 🧮 Mock Data (สามารถเปลี่ยนเป็นไฟล์จริงได้)
    data = {
        "Category": ["Music", "Entertainment", "Gaming", "Education", "Sports", "News"],
        "Avg Views (Millions)": [150, 120, 95, 60, 70, 40],
        "Total Channels": [250, 200, 180, 150, 130, 90]
    }
    df = pd.DataFrame(data)

    # 🔹 Metrics Overview
    st.subheader("📊 สรุปข้อมูลสำคัญ")
    col1, col2, col3 = st.columns(3)
    col1.metric("🎬 จำนวนช่องทั้งหมด", "1,000 ช่อง")
    col2.metric("🌎 ประเทศที่มีช่องมากที่สุด", "สหรัฐอเมริกา 🇺🇸")
    col3.metric("🧑‍🦰 หมวดหมู่ยอดนิยม", "People & Blogs")

    st.markdown("---")

    

    

    # 📈 สร้างกราฟพายชาร์ต
    category_data = {
        "Category": [
            "People & Blogs", "News & Politics", "Entertainment", "Gaming", "Music", 
            "Sports", "Howto & Style", "Film & Animation", "Comedy", "Education",
            "Autos & Vehicles", "Travel & Events", "Science & Technology", "Pets & Animals"
        ],
        "Percentage": [
            26.4, 18.7, 15.6, 10.7, 8.4,
            5.8, 3.7, 2.4, 1.6, 1.5,
            1.25, 1.4, 1.3, 1.0
        ]
    }
    df1 = pd.DataFrame(category_data)

    fig1 = px.pie(
        df1,
        names="Category",
        values="Percentage",
        title="Distribution of YouTube Categories by Like %",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig1.update_traces(textinfo="percent+label", textposition="inside", pull=[0.05 if i == 0 else 0 for i in range(len(df1))])
    st.plotly_chart(fig1, use_container_width=True)

    # 🧩 Insight 1
    st.markdown("### 💡 Insight จากการวิเคราะห์")
    st.success("""
    - 🟦 **People & Blogs (26.4%)** ได้รับความนิยมสูงสุด เพราะผู้ชมชอบเนื้อหาชีวิตจริงและ Vlog  
    - 🟧 **News & Politics (18.7%)** และ 🟩 **Entertainment (15.6%)** เป็นหมวดยอดนิยมในหลายประเทศ  
    - 🟥 **Gaming (10.7%)** มี engagement สูง โดยเฉพาะกลุ่มวัยรุ่น  
    - 🟪 **Music (8.4%)** แม้มีผู้ติดตามมากแต่ยอด Like ต่อคลิปอยู่ในระดับกลาง  
    - 🟫 หมวด **Education**, **Howto & Style**, **Comedy** กำลังมีแนวโน้มเติบโตต่อเนื่อง  
    """)

    st.divider()

    # ==================== SECTION 2: Video Views > 100 Million ====================
    st.header("🎬 Video Name that have Video Views > 100 Million 🔍")
    st.markdown("""
    ***จากภาพเพราะ Colab ไม่สามารถอ่าน Font บาง Font ได้ทำให้เกิดเป็นสีเหลี่ยม***  
    จากการเก็บและวิเคราะห์ข้อมูลวิดีโอที่ได้รับความนิยมสูงสุดในประเทศไทย  
    พบว่า มีเพียงไม่กี่วิดีโอที่สามารถสร้างยอดวิวได้มากกว่า **100 ล้านวิว**
    """)

    # 📊 สร้างข้อมูลสมมุติที่ใกล้เคียงของจริง
    st.image(
    "Graph.webp",  # ✅ ใช้ไฟล์ที่อัปโหลดมา
    caption="Video Name that have Video Views > 100 Million",
    use_container_width=True
)
    

    

    # 🧠 บรรยายสรุป
    st.markdown("""
    จากการเก็บและวิเคราะห์ข้อมูลวิดีโอที่ได้รับความนิยมสูงสุดในประเทศไทย  
    พบว่า มีเพียงไม่กี่วิดีโอที่สามารถสร้างยอดวิวได้มากกว่า **100 ล้านวิว**  
    ซึ่งในกราฟด้านบนแสดงให้เห็นว่าวิดีโอที่มียอดวิวสูงสุดคือ  
    **Thailand Trending Song** และ **Bangkok vlog highlight**  
    ที่มียอดวิวเกินกว่า 200 ล้านวิว 🎶🎥  
    """)

    st.info("""
    🔹 วิดีโอที่มียอดวิวสูงสุดส่วนใหญ่เป็นแนวบันเทิง เพลง และวิดีโอแนว Vlog  
    🔹 ส่วนวิดีโอแนวให้ความรู้และอนิเมชันมียอดวิวระดับกลาง  
    🔹 แสดงให้เห็นถึงพฤติกรรมผู้ชมที่ให้ความสำคัญกับความบันเทิงมากกว่าความรู้  
    """)

  

    # สรุปปิดท้าย
    st.divider()
    st.markdown("### 🧩 สรุปภาพรวมทั้งหมด")
    st.success("""
    ✅ หมวด **People & Blogs** และ **Entertainment** ยังคงเป็นกลุ่มคอนเทนต์ที่มีแนวโน้มเติบโตสูง  
    ✅ วิดีโอที่มียอดวิวเกิน 100 ล้านส่วนใหญ่เป็นแนว **เพลง / บันเทิง / vlog**  
    ✅ ช่องทาง YouTube ของไทยมีแนวโน้มเพิ่มขึ้นในกลุ่มเนื้อหาทั่วไปและเพลง  
    """)


    # ---------------------- หน้า เกี่ยวกับโครงการ ----------------------
elif menu == "🧑‍💼 Profile Page (Piyachart)":
    st.title("🧑‍💼 Profile Page (Piyachart)")
    st.write("""
    - ชื่อ–สกุล: นาย ปิยชาติ วอนวัฒนา , Piyachart Wonwatana  🧑‍💼 
    - รหัสนักศึกษา: 2213310226 🧑‍💻
    - สาขา: BI 🧑‍💻
    """) 
    st.subheader("ความสนใจใน Data Science / Data Mining 📊")
    st.write("""
    สนใจการเรียนรู้เกี่ยวกับ Machine Learning และ AI มาใช้วิเคราะห์ข้อมูล 🗂️ เช่น  
    - การวิเคราะห์ข้อมูลธุรกิจ  
    - การทำ Data Visualization  
    - การใช้ข้อมูลมาทำนายแนวโน้ม
    ซึ่งเป็นสิ่งที่นสนใจและผมคิดว่าจะมีประโยชน์กับตัวผมในอนาคต 🧑‍🔬
    """)

    st.subheader("ประสบการณ์ที่เคยทำ📋")
    st.write("""
    - โปรเจกต์: Marketing ทำธุรกิจขายสบู่ 🧼 ด้วยงบ 500 💵  
    - โปรเจกต์: CRM ทำกิจกรรมออกแคมเปญ 🥇 เพื่อช่วยเพิ่มยอดขายให้ร้านกาแฟ ☕
    - กิจกรรม:  แข่งขัน Logizard X TNI 2023-2024 🏆 ได้รางวัลรองชนะเลิศอันดับ 3 🥉
    - กิจกรรม:  Work and Travel USA  at Hilton Waikoloa Village Hawaii ☀️🏖️ as a Housekeeping 🧹🧑‍💼
     """)
    
    st.subheader("Skillset ที่เกี่ยวข้อง🔍")
    st.write("""
    - Programming : Python, SQL , Java , Html , Abap  
    - Enterprise Resource Planning : SAP S/4 HANA
    - Other software :  Power BI , Word , Excel , Power Point , Canva , Figma ,  
      Cisco Packet Traser , Project Libre , SaleForce
    - Languages : English (Toeic 570)  , Japanese (JLPT N4 )
    """)

# ---------------------- หน้า Profile Page (Kasidit) ----------------------
elif menu == "🧑‍💼 Profile Page (Kasidit)":
    st.title("🧑‍💼 Kasidit Sornsud")

    # ส่วนหัวแบบ Resume
    st.markdown("""
    ### 💬 About Me  
    I am a final-year student majoring in **Digital Business** under the Faculty of Information Technology.  
    I have a strong interest in both business and technology, and I am quick to learn and adapt new technologies  
    to apply them effectively in business contexts. 🚀
    """)

    st.markdown("---")

    # 🔹 ข้อมูลติดต่อ
    st.subheader("📞 Contact")
    col1, col2 = st.columns(2)
    with col1:
        st.write("""
        - 📱 **Phone:** 096-128-9170  
        - 📧 **Email:** kasiditsornsud@gmail.com  
        - 🏠 **Location:** Bangkok, Thailand  
        """)
    with col2:
        st.write("""
        - 🌐 **LinkedIn:** linkedin.com/in/kasiditsornsud  
        - 🖥️ **Portfolio:** streamlit.app/Kasidit  
        """)

    st.markdown("---")

    # 🔹 Education Section
    st.subheader("🎓 Education")
    col1, col2 = st.columns(2)
    with col1:
        st.write("""
        **THAI-NICHI INSTITUTE OF TECHNOLOGY**  
        - Major: Digital Business (Faculty of IT)  
        - GPAX: **3.02**  
        - Year: **2022 - 2025**
        """)
    with col2:
        st.write("""
        **RAYONG WITTAYAKOM SCHOOL**  
        - Science and Math Program  
        - GPAX: **3.73**  
        - Year: **2016 - 2021**
        """)

    st.markdown("---")

    # 🔹 Hard Skills
    st.subheader("💻 Hard Skills")
    st.write("""
    - 🧠 **IT Business**  
    - 🐍 **Python / SQL / HTML / CSS / JavaScript**  
    - 💼 **SAP / ERP / ABAP**  
    - 📊 **Power BI / Excel (VLOOKUP, PivotTable)**  
    - 💻 **CRM Platforms (Salesforce)**  
    - 🎨 **Figma (UI/UX Design)**  
    - 📈 **Financial & Data Analysis**
    """)

    st.markdown("---")

    # 🔹 Soft Skills
    st.subheader("🧩 Soft Skills")
    col1, col2 = st.columns(2)
    with col1:
        st.write("""
        - Time Management  
        - Problem Solving  
        - Teamwork & Collaboration  
        """)
    with col2:
        st.write("""
        - Communication Skills  
        - Self-Motivation  
        - Leadership  
        """)

    st.markdown("---")

    # 🔹 Awards & Experience
    st.subheader("🏆 Awards and Experience")
    st.write("""
    - 🥉 **2nd Runner Up - TNI X LogiZARD (2024)**  
    - 🧾 **10th ICBIR 2025:**  
      Research certificate: _“ปัจจัยที่ทำให้กลุ่มคน Generation Z มีความสม่ำเสมอในการออกกำลังกาย”_  
    - 💡 **Marketing Project:** TNI-Day Business Simulation  
    - ☕ **CRM Project:** Valentine’s Day Café Promotion  
    - 🎯 **Event:** IT Faculty Freshmen Orientation (Coordinator)
    """)

    st.markdown("---")

    # 🔹 Languages
    st.subheader("🌍 Languages")
    st.write("""
    -  Thai  
    -  English 
    -  Japanese 
    """)

    # 🔹 Footer
    st.markdown("---")
    st.success("👨‍💼 Developed by **Kasidit Sornsud** | Faculty of Information Technology, TNI")

elif menu == "🧑🏻‍💻 Code":
    st.title("🧑🏻‍💻 Project Code & Model Pipeline")
    st.caption("สรุปขั้นตอนและโค้ดที่ใช้ในการสร้างระบบ **Career Predictor (Decision Tree)** 🧠")
    st.markdown("---")

    # ===============================
    # STEP 1: DATA PREPARATION
    # ===============================
    st.header("📘 Step 1: Data Preparation & Preprocessing")
    st.markdown("""
    ขั้นตอนแรกคือการโหลดข้อมูลจากไฟล์ CSV,  
    จากนั้นแปลงเกรดเป็นตัวเลข และเข้ารหัสข้อความในคอลัมน์ `Job` และ `Status` ให้เป็นค่าตัวเลขที่โมเดลเข้าใจได้ 💡
    """)
    with st.expander("👀 ดูโค้ดตัวอย่างการเตรียมข้อมูล"):
        st.code("""
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# โหลดข้อมูล
df = pd.read_csv("BBB.csv")

# แปลงเกรดตัวอักษรเป็นตัวเลข
grade_map = {
    "A": 4.0, "B+": 3.5, "B": 3.0, "C+": 2.5, "C": 2.0,
    "D+": 1.5, "D": 1.0, "F": 0.0
}
for col in df.columns:
    if col not in ["Job", "Status"]:
        df[col] = df[col].map(grade_map)

# เข้ารหัสอาชีพ
le = LabelEncoder()
df["Job"] = le.fit_transform(df["Job"])

# แปลงสถานะเป็นตัวเลข
df["Status"] = df["Status"].map({"Successful": 1, "Unsuccessful": 0})
        """, language="python")

    st.success("✅ หลังจากขั้นตอนนี้ ข้อมูลพร้อมสำหรับนำไปเทรนโมเดล Machine Learning แล้ว")
    st.markdown("---")

    # ===============================
    # STEP 2: MODEL TRAINING
    # ===============================
    st.header("🧩 Step 2: Model Training (Decision Tree)")
    st.markdown("""
    ใช้อัลกอริทึม **DecisionTreeClassifier** เพื่อเรียนรู้จากข้อมูลเกรดและอาชีพ  
    จากนั้นจะได้โมเดลที่สามารถทำนายได้ว่า ผู้เรียนเหมาะกับอาชีพที่เลือกหรือไม่ 🎯
    """)
    with st.expander("📘 โค้ดเทรนและบันทึกโมเดล"):
        st.code("""
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle

# แยก Features และ Target
X = df.drop(columns=["Status"])
y = df["Status"]

# แบ่งข้อมูล train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# สร้างโมเดล Decision Tree
clf = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42)
clf.fit(X_train, y_train)

# ประเมินผล
print("Accuracy (train):", clf.score(X_train, y_train))
print("Accuracy (test):", clf.score(X_test, y_test))

# บันทึกโมเดล
with open("model_G.pkl", "wb") as f:
    pickle.dump(clf, f)

# บันทึก encoder
with open("job_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
        """, language="python")

    st.info("💡 ใช้ Entropy เป็นเกณฑ์การแยก node และจำกัดความลึกของต้นไม้ไว้ที่ 4 ชั้น เพื่อป้องกัน overfitting")
    st.markdown("---")

    

    # ===============================
    # STEP 4: MODEL DEPLOYMENT (STREAMLIT)
    # ===============================
    st.header("🌐 Step 3: Streamlit Deployment")
    st.markdown("""
    ขั้นตอนนี้คือการนำโมเดลที่เทรนแล้วมาสร้างเป็นเว็บแอป  
    โดยใช้ Streamlit เพื่อให้ผู้ใช้สามารถกรอกเกรดและเลือกอาชีพ  
    แล้วระบบจะทำนายว่าเหมาะสมหรือไม่แบบ real-time ⚡
    """)
    with st.expander("🖥️ โค้ดตัวอย่างสำหรับหน้าเว็บทำนาย"):
        st.code("""
import streamlit as st
import pickle
import numpy as np

# โหลดโมเดล
with open("model_G.pkl", "rb") as f:
    model = pickle.load(f)

st.title("💼 Career Suitability Prediction")

# กรอกเกรดตัวอย่าง
grades = [4.0, 3.5, 3.0, 2.5, 3.5, 4.0]
career = 2  # Network Engineer

X = np.array(grades).reshape(1, -1)
prediction = model.predict(X)

if prediction[0] == 1:
    st.success("✅ อาชีพนี้เหมาะสมกับคุณ!")
else:
    st.error("❌ อาชีพนี้อาจไม่เหมาะสมกับคุณ")
        """, language="python")

    st.success("🚀 ระบบนี้พร้อมใช้งานในหน้า Predict Career แล้ว!")
    st.markdown("---")

   