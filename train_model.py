import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# 1. โหลดข้อมูล
df = pd.read_csv("BBB.csv")

# 2. แปลงเกรดเป็นตัวเลข
grade_map = {
    "A": 4.0, "B+": 3.5, "B": 3.0, "C+": 2.5, "C": 2.0,
    "D+": 1.5, "D": 1.0, "F": 0.0
}
for col in df.columns:
    if col not in ["Job", "Status"]:
        df[col] = df[col].map(grade_map)

# 3. เข้ารหัสข้อความในคอลัมน์ Job ให้เป็นตัวเลข
le = LabelEncoder()
df["Job"] = le.fit_transform(df["Job"])

# 4. แปลง Status เป็น 1/0
df["Status"] = df["Status"].map({"Successful": 1, "Unsuccessful": 0})

# 5. แยก Features (X) และ Target (y)
X = df.drop(columns=["Status"])   # ✅ ไม่ต้อง drop Job แล้ว
y = df["Status"]

# 6. แบ่ง train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. เทรน Decision Tree
clf = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42)
clf.fit(X_train, y_train)

# 8. ประเมินผล
print("Accuracy (train):", clf.score(X_train, y_train))
print("Accuracy (test):", clf.score(X_test, y_test))

# 9. บันทึกโมเดล
with open("model_G.pkl", "wb") as f:
    pickle.dump(clf, f)

