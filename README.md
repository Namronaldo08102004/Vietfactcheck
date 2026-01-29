# VietFactCheck: A Comprehensive Vietnamese Fact-Checking Pipeline

**VietFactCheck** is an end-to-end framework designed to verify the factual accuracy of Vietnamese claims. The project implements a multi-stage pipeline—ranging from claim detection and document retrieval to final verification—tailored specifically for the linguistic nuances of the Vietnamese language.

## 📂 Project Structure

This repository is organized into four main components.

* **[`data/`](https://www.google.com/search?q=./data/)**: Contains all datasets used throughout the project. This includes raw and pre-processed data for **Extraction** (Claim Detection), **Retrieval**, and **Verification** modules. It also houses our custom-generated **ViNewsCheck** dataset.
* **[`experiments/`](https://www.google.com/search?q=./experiments/)**: This is our research laboratory. We have conducted extensive testing here, covering not only individual modules (like Claim Extraction or Evidence Selection) but also **Combined Pipelines** (e.g., `DC+ES+CV`, `DC+CV`). These experiments help determine the most effective end-to-end configurations for the fact-checking task. Detailed results and analyses can be found in `Analysis.ipynb`.
* **[`src/`](https://www.google.com/search?q=./src/)**: The core source code of the application. It contains the modular implementation of the pipeline, vector database configurations for document retrieval, and the main logic for the user interface.
* **[`report/`](https://www.google.com/search?q=./report/)**: Technical reports, presentation slides, and comprehensive research documentation detailing our methodology and findings.

---

## 🔗 External Links

We provide public access to our models and demonstrations to support reproducibility in the Vietnamese NLP community:

* 🧠 **Model Collection**: [VietFactCheck PLMs on Hugging Face](https://huggingface.co/collections/Namronaldo2004/vifactcheck-plm) – A collection of Pre-trained Language Models fine-tuned for various stages of the pipeline.
* 📺 **Video Demo**: [Watch the demo](https://www.google.com/search?q=%23) *(Coming soon)*

---

## 🚀 Getting Started

Follow these steps to set up and run the VietFactCheck WebApp on your local machine.

### 1. Installation

Install the required Python dependencies:

```bash
pip install -r requirements.txt

```

### 2. Prerequisites (Java)

The project utilizes the **VnCoreNLP** toolkit for Vietnamese language processing.

> [!IMPORTANT]
> You must have **Java Runtime Environment (JRE) >= 1.8** installed on your system. You can verify your version by running:
> ```bash
> java -version
> 
> ```
> 
> 

### 3. Running the App

Launch the interactive web interface using Streamlit:

```bash
streamlit run src/app.py

```

---

## 🧑‍💻 Contributors

This project was developed with passion by:

* [Namronaldo08102004](https://github.com/Namronaldo08102004)
* [CISTILY](https://github.com/CISTILY)

## 🙏 Acknowledgements

We would like to express our sincere gratitude to:

* **Dr. Nguyen Truong Son** and **Dr. Nguyen Tien Huy** for their invaluable guidance, expert insights, and constant support throughout this research.
* **HCMUS – University of Science, VNU-HCM**, for providing the academic foundation and resources necessary to bring this project to life.