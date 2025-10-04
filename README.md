<img width="923" height="857" alt="image" src="https://github.com/user-attachments/assets/25ace725-94ab-48f4-9462-b942eba1fcd2" /># 🔬 Skin Disease Classification AI

A user-friendly web application powered by a deep learning model to classify skin diseases from images. This project aims to provide an accessible tool for preliminary skin condition analysis.

**[Live Demo](https://skin-disease-classifier.netlify.app/)** &nbsp;&middot;&nbsp;

![Website Screenshot](https://github.com/pm-ju/skin-disease-classification/blob/main/Screenshot%202025-10-04%20122458.png )
![Predictions](https://github.com/pm-ju/skin-disease-classification/blob/main/Screenshot%202025-10-04%20122450.png)
*(**Note:** Replace the placeholder above with a real screenshot of your web application)*

---

##  Table of Contents

-   [About The Project](#-about-the-project)
-   [Key Features](#-key-features)
-   [Technology Stack](#-technology-stack)
-   [Dataset Information](#-dataset-information)
-   [Getting Started](#-getting-started)
    -   [Prerequisites](#prerequisites)
    -   [Installation & Setup](#installation--setup)
    -   [Running the Application](#running-the-application)
-   [Project Structure](#-project-structure)
-   [Contributing](#-contributing)
-   [License](#-license)


---

##  About The Project

This project combines a robust Convolutional Neural Network (CNN) with a clean web interface to classify various types of skin lesions. Users can upload an image of a skin lesion, and the AI model will provide a prediction for one of several diagnostic categories.

**Medical Disclaimer:** This is an academic and experimental project. The predictions made by this application are for informational purposes only and are **not a substitute for professional medical advice, diagnosis, or treatment**. Always seek the advice of a qualified dermatologist or healthcare provider with any questions you may have regarding a medical condition.

---

##  Key Features

* **AI-Powered Classification:** Leverages a deep learning model for lesion classification.
* **Simple Web Interface:** Intuitive file upload functionality.
* **Instant Results:** Provides a classification prediction and a confidence score.
* **Modular Code:** Includes a `model_loader.py` for cleanly loading the trained model.

---

##  Technology Stack

* **Backend:** Python, Flask
* **Machine Learning:** PyTorch, Scikit-learn, Pandas, NumPy
* **Frontend:** HTML5, CSS3, JavaScript

---

##  Dataset Information

The model is trained on the **ISIC 2018: Skin Lesion Analysis Towards Melanoma Detection** challenge dataset ("Task 3: Lesion Diagnosis").

To train the model or run the application with full context, you will need to download the following components into a local `data/` directory (which you should create):

1.  **Image Data:** The primary dataset of skin lesion images.
    * **Source:** [ISIC 2018 Challenge Data](https://challenge.isic-archive.com/data/#2018)
    * **Required File:** `ISIC2018_Task3_Training_Input.zip`
        ![ISIC 2018 Dataset Download](https://github.com/user-attachments/assets/ded26b9a-1150-48a2-8ee2-705a8367cef8)

2.  **Metadata File:** Contains labels and other information for each image.
    * **Source:** [metadata.csv](https://raw.githubusercontent.com/tkalbl/RevisitingSkinToneFairness/main/data/ISIC2018/ISIC2018_Task3_Training_Input/metadata.csv)

3.  **Corrected Metadata:** An additional file with corrected data points.
    * **Source:** [Bevan_corrected.csv](https://raw.githubusercontent.com/tkalbl/RevisitingSkinToneFairness/main/Bevan_corrected.csv)

---

##  Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

* Python 3.8+ and `pip`
* Git

### Installation & Setup

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-github-username/skin-disease-classification.git](https://github.com/your-github-username/skin-disease-classification.git)
    cd skin-disease-classification
    ```

2.  **Set up Python Virtual Environment (Recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the Model:**
    Make sure your trained model file (e.g., `model.h5`) is placed inside the `models/` directory. If it's stored elsewhere (like Git LFS or a cloud drive), download it into that folder.

### Running the Application

1.  **Start the Backend Server:**
    ```bash
    python app.py
    ```

2.  **Access the Web App:**
    Open your web browser and navigate to `http://127.0.0.1:5000` (or the port specified in your `app.py`).

---

##  Project Structure

This repository has the following structure:
```bash
skin-disease-classification/
├── models/               # Contains the saved deep learning model
├── app.py                # The main Flask server application
├── model_loader.py       # Utility script to load the ML model
├── requirements.txt      # List of Python dependencies for pip
├── LICENSE               # Project license file (MIT)
└── README.md             # This README file
```
---

##  Contributing

Contributions are what make the open-source community an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

---

##  License

Distributed under the **MIT License**. See `LICENSE` for more information.

---
