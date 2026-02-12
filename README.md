# AI-Based Team Compatibility Predictor 🚀

An AI-driven framework for **intelligent team formation** in software projects by jointly modeling **technical skills, personality traits, and interpersonal compatibility**, with **real-time behavioral monitoring** during project execution.

## 📌 Motivation

Traditional team formation methods emphasize **technical skills alone**, often ignoring **human behavioral compatibility**, leading to:
- Interpersonal conflicts  
- Poor collaboration  
- Reduced project efficiency  

Research shows that **team success depends heavily on personality alignment and communication patterns**, not just expertise.  
This project addresses that gap using **Machine Learning, NLP, and Generative AI**.

---

## 🧠 Key Contributions

- ✅ Combines **technical skill matching + Big-Five personality traits**
- ✅ Uses **Generative AI (LLM)** to auto-generate project roles and team size
- ✅ Computes **pairwise and team-level compatibility scores**
- ✅ Supports **real-time behavioral monitoring** using Slack & Trello
- ✅ Scalable **beam-search-based team formation**
- ✅ Human-centric and data-driven framework

---

## 🏗️ System Architecture

The system follows a **multi-stage pipeline**:

1. **Project Understanding (LLM-based)**
2. **Candidate Profiling**
3. **Technical Skill Scoring**
4. **Personality Trait Inference (Big-Five)**
5. **Compatibility Score Calculation**
6. **Team Formation using Beam Search**
7. **Real-Time Behavioral Monitoring**
8. **Adaptive Team Analytics**

---

## 📂 Repository Structure
AI-Based-Team-Compatibility-Predictor/
│
├── app.py                     # Main Flask application (routes + backend logic)
├── app1.py                    # Secondary/experimental Flask app (testing or alternate flow)
│
├── aisubmodule.py              # LLM-based project role & team size generator
├── technicalscore.py           # Technical skill scoring mechanism
├── behavioral_score.py         # Behavioral score computation logic
├── compatibilitycheck.py       # Pairwise & team compatibility calculations
├── trainedData.py              # Model loading & inference utilities
│
├── behavioral_model.pkl        # Fine-tuned RoBERTa personality model
│
├── data/
│   ├── TrainingData.xlsx       # Personality training dataset
│   ├── UpdatedDataset.xlsx     # Employee profiling dataset
│
├── templates/                  # HTML templates (Flask frontend)
│   ├── index.html              # Landing page
│   ├── teamformation.html      # Team formation input & results page
│   ├── dashboard.html          # Compatibility scores & analytics dashboard
│   └── monitor.html            # Real-time behavioral monitoring page
│
├── static/                     # Static assets
│   ├── css/
│   ├── js/
│   └── images/
│
├── requirements.txt            # Python dependencies
├── venv/                       # Virtual environment (not pushed to GitHub)
│
└── README.md                   # Project documentation



---

## 🤖 AI Submodule (Project Understanding)

- Uses **Mistral-7B-Instruct**
- Inputs:
  - Project title
  - Project description
- Outputs:
  - Recommended team size
  - Role definitions
  - Required skills per role
- Reduces manual effort in project planning

---

## 🧪 Technical Skill Scoring

Each employee is scored against role requirements:

\[
\text{Technical Score} = 100 \times \frac{\text{Matched Skills}}{\text{Required Skills}}
\]

Top-*k* candidates per role move to behavioral analysis.

---

## 🧠 Personality Trait Inference

- Uses **Big-Five (OCEAN) model**
  - Openness
  - Conscientiousness
  - Extraversion
  - Agreeableness
  - Neuroticism
- Behavioral summaries processed using a **fine-tuned RoBERTa model**
- Multi-task learning with **5 trait-specific heads**

### Overall Behavioral Score:
\[
\frac{1}{5}(O + C + E + A + (1 - N))
\]

---

## 🔗 Compatibility Score Calculation

### Similarity-based traits:
- Conscientiousness
- Agreeableness

\[
1 - |Trait_A - Trait_B|
\]

### Complementary traits:
- Openness
- Extraversion
- Neuroticism

\[
1 - |(Trait_A + Trait_B) - 1|
\]

### Team Compatibility:
\[
\frac{\sum \text{Pairwise Scores}}{\text{Total Pairs}} \times 100
\]

---

## ⚡ Team Formation Strategy

- Uses **Greedy Beam Search**
- Avoids exponential brute-force combinations
- Achieves:
  - ~70% reduction in runtime
  - Near-optimal team quality

---

## 📊 Real-Time Behavioral Monitoring

### 🔹 Slack Integration
- Monitors:
  - Message frequency
  - Sentiment trends
  - Response time
  - Conflict indicators
- Updates personality traits dynamically using rolling message windows

### 🔹 Trello Integration
- Tracks:
  - Task completion
  - Delay frequency
  - Workload balance

### Adaptive Behavioral Score:
\[
B^*(t) = \beta B(t) + (1 - \beta) T(t)
\]

---

## 📈 Experimental Results

| Metric | Traditional | Proposed |
|------|------------|----------|
| Technical Fit | ~72% | **85% (+13%)** |
| Team Compatibility | 63% | **79% (+16%)** |
| Runtime | 220s | **65s (70% faster)** |
| Real-Time Adaptability | ❌ | ✅ |

---

## 🏁 Conclusion

This project demonstrates that:
- **Personality-aware team formation significantly improves collaboration**
- **Moderate personality prediction accuracy is sufficient for team-level gains**
- **AI can enable adaptive, human-centric team management**

The framework is suitable for:
- Software companies  
- Academic project teams  
- HR analytics platforms  

---

## 🔮 Future Work

- Feedback-driven learning from completed projects
- Cross-team optimization
- Deployment in multi-organizational environments
- Dashboard for team health visualization

---

## 📜 Citation

If you use this work, please cite:


---

## 🤝 Contributors

- **Joshitha Chennamsetty**
- Venkata Vaishnavi Uppiretla  
- Ravi Varma Chakrala  
- Bharath Chandra Sai Sakhamuri  
- Prof. K. L. V. G. Krishna Murthy  

---

## ⭐ Acknowledgements

- IEEE  
- Hugging Face Transformers  
- Mistral AI  
- Slack & Trello APIs  

---

⭐ *If you find this project useful, please consider starring the repository!*  

