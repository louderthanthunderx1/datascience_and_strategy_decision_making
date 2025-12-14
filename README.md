# 📊 Polynomial Regression Assignment

> **วิชา:** Data Science and Strategy Decision Making  
> **Assignment:** 1 - Polynomial Regression with Least Squares and Ridge Regression

---

## 👤 ข้อมูลนักศึกษา (Student Information)

| Parameter | ค่า | ที่มา |
|-----------|-----|-------|
| **k** | 7 | จำนวนตัวอักษรในนามสกุล "sukchok" |
| **m** | 3 | จำนวนตัวอักษรในชื่อเล่น "lok" |

---

## 📖 คำอธิบายโจทย์ (Problem Description)

### 🎯 เป้าหมาย
ประมาณค่า (Estimate) แบบจำลองพหุนาม (Polynomial Model) จากข้อมูล Training ที่มี Noise แล้วพล็อตกราฟเปรียบเทียบระหว่างข้อมูลดิบกับโมเดลที่ประมาณได้

### 📊 ข้อมูล Training Data
- **ช่วงของ x:** `[0.1, k]` → `[0.1, 7]`
- **Step size:** `0.0m` → `0.03`
- **จำนวนจุดข้อมูล:** 30 จุด
- **Noise:** Zero-mean Gaussian noise (ค่าเฉลี่ย = 0)

### 🧮 แบบจำลองพหุนาม (Polynomial Model)

สมการพหุนามทั่วไป degree n:

```
y = a₁x + a₂x² + a₃x³ + ... + aₙxⁿ + b
```

### 📐 ระบบสมการ Overdetermined

เมื่อมีข้อมูล 30 จุด และใช้พหุนาม degree = n จะได้ระบบสมการที่มีสมการมากกว่าตัวแปร:

```
┌                      ┐ ┌    ┐   ┌    ┐
│ x₁   x₁²  ... x₁ⁿ  1 │ │ a₁ │   │ y₁ │
│ x₂   x₂²  ... x₂ⁿ  1 │ │ a₂ │ = │ y₂ │
│ :    :    ... :    : │ │ :  │   │ :  │
│ x₃₀  x₃₀² ... x₃₀ⁿ 1 │ │ aₙ │   │y₃₀ │
└                      ┘ │ b  │   └    ┘
                         └    ┘
```

เขียนย่อๆ: **AX = B**

---

## 🔬 วิธีการแก้ปัญหา (Solution Methods)

### 1️⃣ Least Squares Solution

หาค่า X ที่ดีที่สุดโดยใช้ **Cost Function** ในรูปแบบ Linear Least Squares:

```
X* = (AᵀA)⁻¹AᵀB = A†B
```

โดยที่:
- **Aᵀ** = Transpose ของ matrix A
- **A†** = Pseudo-inverse ของ matrix A
- สูตรนี้หา coefficients ที่ทำให้ค่า error กำลังสองน้อยที่สุด

### 2️⃣ Ridge Regression

เพิ่ม **Regularization Term** เพื่อป้องกัน Overfitting:

```
X^Ridge = (AᵀA + ρI)⁻¹AᵀB
```

โดยที่:
- **ρ (rho)** = Regularization parameter (ค่า penalty)
- **I** = Identity matrix
- ρ ยิ่งมาก → coefficients ยิ่งถูกทำให้เล็กลง → เส้นเรียบขึ้น

---

## 📝 Tasks และคะแนน

| Task | คำอธิบาย | คะแนน |
|------|---------|-------|
| **1** | Polynomial Regression degree 1 (เส้นตรง) | 20 คะแนน |
| **2** | Polynomial Regression degree 3 | 20 คะแนน |
| **3** | Polynomial Regression degree 7 | 30 คะแนน |
| **4.1** | Ridge Regression degree 7 โดย ρ = 10⁻⁶ | 15 คะแนน |
| **4.2** | Ridge Regression degree 7 โดย ρ = 0.1 | 15 คะแนน |
| | **รวม** | **100 คะแนน** |

---

## 💡 แนวคิดสำคัญ (Key Concepts)

### Underfitting vs Overfitting

| สถานการณ์ | ปัญหา | ตัวอย่าง |
|-----------|-------|---------|
| **Underfitting** | Model เรียบเกินไป จับ pattern ไม่ได้ | Degree 1 (เส้นตรง) |
| **Overfitting** | Model จับ noise แทนที่จะจับ pattern จริง | Degree 7 (ไม่มี regularization) |

### Ridge Regression ช่วยอย่างไร?

- **ρ เล็ก (10⁻⁶):** ผลใกล้เคียงกับ Least Squares ปกติ
- **ρ ใหญ่ (0.1):** Coefficients ถูกบังคับให้เล็กลง → เส้นเรียบขึ้น → ลด Overfitting

---

## 📁 โครงสร้างโปรเจค (Project Structure)

```
code/
├── main.py              # โปรแกรมหลัก - รันทุก Tasks
├── pyproject.toml       # Dependencies
├── README.md            # เอกสารอธิบาย (ไฟล์นี้)
├── src/
│   ├── generate.py      # สร้าง Training Data
│   ├── metrics.py       # สร้าง Data Matrix และ Least Squares
│   ├── solver.py        # Ridge Regression และ Prediction
│   └── plot.py          # พล็อตกราฟ
└── output/
    ├── task1_degree1.png
    ├── task2_degree3.png
    ├── task3_degree7.png
    ├── task4_1_ridge_1e-6.png
    ├── task4_2_ridge_0.1.png
    └── comparison_all_methods.png
```

---

## 🔧 วิธีการติดตั้งและรัน (Installation & Usage)

### ติดตั้ง Dependencies

```bash
# ใช้ uv (แนะนำ)
uv sync

# หรือใช้ pip
pip install numpy matplotlib
```

### รันโปรแกรม

```bash
# ใช้ uv
uv run main.py

# หรือใช้ python โดยตรง
python main.py
```

---

## 📊 Output ที่ได้

### กราฟแต่ละ Task
แต่ละ Task จะแสดงกราฟที่มี:
- 🔴 **จุดสีแดง:** Noisy Training Data (ข้อมูลดิบ)
- 🔵 **เส้นสีน้ำเงิน:** Estimated Polynomial (เส้นโค้งที่ประมาณได้)

### กราฟเปรียบเทียบทุกวิธี (comparison_all_methods.png)

| สี | วิธี |
|----|------|
| 🔴 จุดแดง | Noisy Training Data |
| 🔵 Cyan | Degree 1 (Linear) |
| 🟠 Orange | Degree 3 |
| 🔵 Blue | Degree 7 (Least Squares) |
| 🟢 Green ประ | Degree 7 Ridge (ρ = 10⁻⁶) |
| 🟣 Magenta ประ | Degree 7 Ridge (ρ = 0.1) |

---

## 📚 อ้างอิงสูตรคณิตศาสตร์ (Mathematical Reference)

### Data Matrix A (สำหรับ degree n)

```python
A = [[x₁,   x₁²,  x₁³,  ..., x₁ⁿ,  1],
     [x₂,   x₂²,  x₂³,  ..., x₂ⁿ,  1],
     ...
     [x₃₀,  x₃₀², x₃₀³, ..., x₃₀ⁿ, 1]]
```

### Least Squares Solution

```
X* = (AᵀA)⁻¹AᵀB
```

### Ridge Regression Solution

```
X^Ridge = (AᵀA + ρI)⁻¹AᵀB
```

### Polynomial Prediction

```
ŷ = a₁x + a₂x² + a₃x³ + ... + aₙxⁿ + b
```

---

## 🎓 สรุป (Summary)

โปรเจคนี้แสดงให้เห็นว่า:

1. **Degree ต่ำ** → Underfitting (ไม่สามารถจับ pattern ได้)
2. **Degree สูง** → อาจ Overfitting (จับ noise แทน pattern)
3. **Ridge Regression** → ช่วยลด Overfitting โดยเพิ่ม penalty term
4. **ρ ที่เหมาะสม** → สมดุลระหว่าง Bias และ Variance

---

*สร้างโดย: sukchok (lok)*
