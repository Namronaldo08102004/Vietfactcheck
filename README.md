# VietFactCheck: A Comprehensive Vietnamese Fact-Checking Pipeline

**VietFactCheck** is an end-to-end framework designed to verify the factual accuracy of Vietnamese claims. The project implements a multi-stage pipeline - ranging from claim detection and document retrieval to final verification - tailored specifically for the linguistic nuances of the Vietnamese language.

## 📂 Project Structure

This repository is organized into four main components. For detailed documentation on each part, please refer to the `README.md` file within the respective folders.

* **[`data/`](./data/)**: Contains all datasets used throughout the project, including raw and pre-processed data for **Extraction**, **Retrieval**, and **Verification**. It also houses our custom-generated **ViNewsCheck** dataset.
* **[`experiments/`](./experiments/)**: Our research laboratory. We conducted extensive testing on individual modules and **Combined Pipelines** (e.g., `DC+ES+CV`, `DC+CV`). Detailed results and analyses are available in `Analysis.ipynb`.
* **[`src/`](./src/)**: The core source code. It contains the modular implementation of the pipeline, vector database configurations, and the main logic for the user interface.
* **[`report/`](./report/)**: Technical reports, presentation slides, and comprehensive research documentation.

## 🔗 External Links

* 🧠 **Model Collection**: [VietFactCheck PLMs on Hugging Face](https://huggingface.co/collections/Namronaldo2004/vifactcheck-plm) – Fine-tuned models for various pipeline stages.
* 📺 **Video Demo**: [Watch the demo]([https://www.google.com/search?q=%23](https://www.youtube.com/watch?v=SSNPT3oVPCI))

## 🚀 Getting Started

Follow these steps to set up and run the VietFactCheck WebApp on your local machine.

### 1. Installation

Install the required Python dependencies:

```bash
pip install -r requirements.txt

```

### 2. Model Weights (Claim Detection)

To run the **Claim Detection** module, you need the **BERTSum** model weights.

1. **Download** the weights from this [Google Drive Folder](https://drive.google.com/drive/u/0/folders/1WOkgwpu2gOnwBrSqM3ci-ayp6ZgKtajU).
2. **Configure the path**: Open the file `src/settings/settings.py` and update the `EXTRACTOR_MODEL_PATH` variable with the local path to your downloaded model file:
```python
# Example in src/settings/settings.py
EXTRACTOR_MODEL_PATH = "path/to/your/downloaded/bertsum_weights.pt"

```

### 3. Prerequisites (Java)

The project utilizes the **VnCoreNLP** toolkit.

> [!IMPORTANT]
> You must have **Java Runtime Environment (JRE) >= 1.8** installed. Verify your version:
> ```bash
> java -version
> 
> ```
> 
> 

### 4. Running the App

Launch the interactive interface using Streamlit:

```bash
streamlit run src/app.py

```

## 🧑‍💻 Contributors

* [Namronaldo08102004](https://github.com/Namronaldo08102004)
* [CISTILY](https://github.com/CISTILY)

## 🙏 Acknowledgements

We would like to express our sincere gratitude to:

* **Dr. Nguyen Truong Son** and **Dr. Nguyen Tien Huy** for their invaluable guidance and expert insights throughout this research.
* **HCMUS – University of Science, VNU-HCM**, for providing the academic foundation and resources.
