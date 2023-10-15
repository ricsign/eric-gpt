# Eric GPT Guide 📖🚀

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ricsign/eric-gpt.git
   cd eric-gpt
   ```

2. **Install Dependencies**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Run Eric GPT**:
   ```bash
   chainlit run python-code-runner.py -w
   ```

## Usage

### main.py
The `main.py` script facilitates an interactive conversation with GPT. Hyperparameters are configurable.

### documents-qna.py
Use `documents-qna.py` to upload a PDF or text file. Eric GPT will then utilize stored embeddings in the vector database for document-based interactions.

### shell.py
With `shell.py`, you can execute shell commands, allowing local file manipulation.

### human-as-a-tool.py
This script employs Langchain's HAAT functionalities, bridging GPT and human input for a collaborative interaction.

### python-code-runner.py
The `python-code-runner.py` script enables Eric GPT to execute Python code within the Replit playground, providing a dynamic coding environment.

## Troubleshooting
1. Ensure all dependencies are installed: `pip install -r requirements.txt`.
2. Update to the latest Python version.
3. Regularly update Eric GPT for the latest features and fixes.
4. If issues persist, consider raising an issue on the GitHub repository.

Thank you for using Eric GPT! 🚀🚀