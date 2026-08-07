# NeuroAI: Intelligent Neurodevelopmental Assessment Platform

## Overview

NeuroAI is a machine learning-based web application designed to assist in the early assessment of neurodevelopmental conditions through behavioral screening and predictive analytics. The platform provides an interactive interface for collecting assessment data, processing it using trained machine learning models, and generating prediction results.

The project aims to support early identification by providing an efficient, data-driven screening system that can be further extended for use by parents, educators, healthcare professionals, and researchers.

---

## Key Features

- Machine learning-based neurodevelopmental risk assessment
- Interactive behavioral screening questionnaire
- Real-time prediction and classification
- Web-based user interface
- SQLite database integration
- Data collection and management
- Prediction history and result storage
- Modular architecture for future feature expansion

---

## Technology Stack

### Programming Language
- Python

### Backend
- Flask

### Frontend
- HTML
- CSS
- JavaScript

### Machine Learning
- Scikit-learn
- Pandas
- NumPy
- Joblib

### Database
- SQLite

### Development Tools
- Visual Studio Code
- Git
- GitHub

---

## Project Structure

```
NeuroAI_Project/
│
├── models/
│   ├── asd_model.pkl
│   ├── neuro_db.py
│
├── static/
├── templates/
│
├── app.py
├── neuroai.html
├── requirements.txt
├── README.md
├── .gitignore
└── datasets/
```

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/srichandana498/NeuroAI.git
```

### Navigate to the Project Directory

```bash
cd NeuroAI
```

### Create a Virtual Environment

```bash
python -m venv venv
```

### Activate the Virtual Environment

**Windows**

```bash
venv\Scripts\activate
```

**Linux/macOS**

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
python app.py
```

Open the application in your browser:

```
http://127.0.0.1:5000
```

---

## System Workflow

```
User Input
      │
      ▼
Behavioral Assessment
      │
      ▼
Data Preprocessing
      │
      ▼
Machine Learning Model
      │
      ▼
Prediction Engine
      │
      ▼
Risk Assessment
      │
      ▼
Result Presentation
```

---

## Future Enhancements

- ADHD assessment
- Dyslexia assessment
- Anxiety and depression screening
- Emotion recognition using computer vision
- Speech-based behavioral analysis
- AI-powered recommendation system
- Parent dashboard
- Teacher dashboard
- Healthcare professional dashboard
- PDF report generation
- Cloud deployment
- Mobile application
- Multi-language support

---

## Screenshots

Project screenshots can be added in an `images` directory and referenced below.

```markdown
![Home Page](images/homepage.png)

![Prediction Dashboard](images/dashboard.png)
```

---

## Applications

This platform can be extended for use in:

- Educational institutions
- Healthcare organizations
- Research laboratories
- Early intervention programs
- Neurodevelopmental screening initiatives

---

## Future Scope

NeuroAI is designed as a scalable platform that can evolve into a comprehensive neurodevelopmental assessment ecosystem by integrating advanced artificial intelligence, multimodal data analysis, cloud infrastructure, and personalized decision-support systems.

---

## Author

**Srichandan**

GitHub: https://github.com/srichandana498

---

## License

This project is intended for academic and educational purposes. Additional licensing information may be added based on future development.
