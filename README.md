# Teaching

**Most people can simply download the provided HTML files in the `notebooks/` folder and use them as-is**—no setup or installations required.

A lightweight, reusable scaffold for teaching materials—Jupyter notebooks and supporting docs—organized for workshops or university courses.

---

## ✨ What’s inside

```

.
├── notebooks/        # Lecture / lab notebooks (Jupyter)
├── templates/        # Ready-to-use HTML handouts / slides
├── CONTRIBUTING.md   # How to contribute
├── LICENSE           # Project license
└── README.md

````

---

## 🚀 Quickstart

### Option A: Use the HTMLs (no install)
1. Open the `notebooks/` folder.
2. Download the HTML file(s) you need.
3. Double-click to open in your browser or drop into your LMS/website.

> Tip: You can lightly customize text and images by opening an HTML file in any editor and saving your changes—still no build step required.

### Option B: Run the notebooks (interactive)
**Clone**
   ```bash
   git clone https://github.com/HaasCP/Teaching.git
   cd Teaching
   ```

Open the files under `notebooks/`.

---

## 🧭 How to use this repo

* Keep each lesson/lab as a separate notebook in `notebooks/`.
* Store reusable materials in `templates/`.
* Add a `data/` folder for datasets.
* Use relative paths so notebooks run across machines.

---

## 🤝 Contributing

See **CONTRIBUTING.md** for guidelines on proposing changes, style conventions, and adding new teaching units.

---

## 📄 License

This project is licensed as described in the **LICENSE** file.

---

## 🗂️ Available notebooks

* **Title:** *“Van Deemter in Practice.”*
* **Audience:** *Students and practitioners in analytical chemistry, chromatography, or method development.*
* **Prereqs:** *None.*
* **Learning outcomes:**

  * LO1 Interpret the A, B/u, and C·u terms and explain how each dominates in different u-regimes.
  * LO2 Identify the optimal flow rate uopt and corresponding Hmin, and relate H to plate number (N = L/H) and resolution (Rs).
  * LO3 Compare particle size effects on A and C, and reason about trade-offs among speed, resolution, and pressure limits.
  * L04 Make design decisions (e.g., choose u, L, and particle size) under realistic constraints such as pressure and extra-column broadening.
* **Estimated time:** *25–40 minutes.*

```
::contentReference[oaicite:0]{index=0}
```
