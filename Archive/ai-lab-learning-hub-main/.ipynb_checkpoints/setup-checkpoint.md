# Setup

For this course, follow one of the two paths below to run the Jupyter Notebooks:
- **Path 1: Local Installation** — for users with Python installed on their laptops.
- **Path 2: Running with EAP** — for users who do not have Python installed locally and have been enabled with EAP previously.

If neither of these apply to you (for example if you don't have Python on your laptop and did not answer the poll previously), please reach out to [kmar@genmab.com](mailto:kmar@genmab.com) ASAP to be enabled for the course.

**IMPORTANT: Before starting the course please complete all of the below steps and ensure that you can successfully run the first cell of the [Introduction Notebook](Introduction/Introduction_to_API_and_Prompting.ipynb). If you have any issues, reach out to the AI Labs team before the course begins.**

---

## Path 1: Local Installation

### Prerequisites

- **VS Code Setup for Jupyter Notebooks:**
  - *Note: We recommend VS Code and will support users with setup, but advanced users can use any IDE or method for running Jupyter Notebooks.*
  - **Download and Install VS Code:**  
    - VS Code is available in the Company Portal. Install it if you haven't already.
  - **Required Extensions for Jupyter Notebooks in VS Code:**
    - **Python Extension (by Microsoft):** Provides Python language support.
    - **Jupyter Extension (by Microsoft):** Enables running and interacting with Jupyter Notebooks.
    - **To Install Extensions:**
      - In VS Code click on the **Extensions** icon in the Activity Bar.
      - Search for “Python” and “Jupyter” in the Extensions Marketplace and select the appropriate extension.
      - Click **Install** on each extension.

 - **uv Installation:**
   - Ensure `uv` is installed on your system. If not, install it following the instructions in the [official documentation](https://docs.astral.sh/uv/getting-started/installation/) for your operating system.
    - For macOS users with Homebrew installed, run:
      ```bash
      brew install uv
      ```
    - For Windows users install with pip:
      ```bash
      pip install uv
      ```

- **Importing the Course Repository:**
  - By default permissions on this repository are read-only and the repository cannot be cloned.
  - Check Your Gitlab Access Level: To see what access level you have for the repository, open the project on GitLab and navigate to the [Members section](https://gitlab.com/genmab/production/non-gxp/ai-labs/ai-lab-learning-hub/-/project_members). Your role (e.g., Guest, Reporter) will be displayed there. 
  - "Guest" or "Custom-Guest" roles:
     - Download the full repository from this [OneDrive folder](https://microsoftgenmab-my.sharepoint.com/:f:/g/personal/kmar_genmab_com/Ejr0aXnsO1pBrTQYbkevO4cBLUew5fr9hiVpV4h6egi1Sw?e=ZI7zVZ).
  - All other roles (e.g. Reporter):
     - Download the full repository from [GitLab](https://gitlab.com/genmab/production/non-gxp/ai-labs/ai-lab-learning-hub/) (select **Code > Download Source Code > zip**).
  - Unzip the file and place the full directory in your desired location.

*After completing the prerequisites, please follow the [Common Steps for All Setups](#common-steps-for-all-setups).*

---

## Path 2: Running with EAP

- **Network Requirement:**
  - EAP is only accessible with VPN or on office networks (preferred, since the VPN can sometimes boot users off unexpectedly).

- **Accessing the Environment:**
  - Go to [https://workbench.qa.eap.genmab.net/](https://workbench.qa.eap.genmab.net/) and sign in with SAML.
  - Click **+ New Session** and select a VS Code session.
    - **Resource Profile:** Medium (recommended)
    - **Image:** `rstudio-r:4.4.1-py3.12.6-beaver`
  - Click **Start Session**. Sessions usually start up pretty quickly, but in some cases it may take a couple of minutes.

- **Setting Up Your VS Code Session:**
  - Once the VS Code session starts, go to the Activity Bar on the left-hand side and select the Explorer (the top icon).
  - Click **Open Folder** and then click **OK** to accept the default home directory location chosen for you.
  - You will now see your home directory folder on the left-hand side of your screen.
  - Click “Yes, I trust the authors” if prompted.

- **Importing the Course Repository:**
  - Check Your Gitlab Access Level: To see what access level you have for the repository, open the project on GitLab and navigate to the [Members section](https://gitlab.com/genmab/production/non-gxp/ai-labs/ai-lab-learning-hub/-/project_members). Your role (e.g., Guest, Reporter) will be displayed there. 
  - "Guest" or "Custom-Guest" roles:
     - Download the full repository from this [OneDrive folder](https://microsoftgenmab-my.sharepoint.com/:f:/g/personal/kmar_genmab_com/Ejr0aXnsO1pBrTQYbkevO4cBLUew5fr9hiVpV4h6egi1Sw?e=ZI7zVZ).
  - All other roles (e.g. Reporter):
     - Download the full repository from [GitLab](https://gitlab.com/genmab/production/non-gxp/ai-labs/ai-lab-learning-hub/) (select **Code > Download Source Code > zip**).
  - Unzip the file.
  - Drag and drop the entire repository contents into the Explorer in your EAP VS Code session.

*After setting up your EAP environment, please follow the [Common Steps for All Setups](#common-steps-for-all-setups).*

---

## Common Steps for All Setups

Follow these steps to install required dependencies for all modules in a virtual environment and configure a Jupyter Notebook in VS Code to use these dependencies.
1. **Open a Terminal Session in VS Code:**
   - Launch VS Code.
   - Open the terminal by clicking **View > Terminal**. (Note: In EAP this is accessed via the hamburger button icon in the upper left hand side of your screen)
   - Navigate to the home directory of the repository if you are not already there.

2. **Synchronize Dependencies**:
   - Run the following command to synchronize to the dependencies specified in the `pyproject.toml` file:
     ```bash
     uv sync
     ```

3. **Activate the Virtual Environment**:
   - After synchronization, activate the virtual environment by running the below in the terminal:
    - For Mac users:
      ```bash
      source .venv/bin/activate
      ```
    - For Windows users:
      ```bash
      .\.venv\Scripts\activate
      ```

4. **Open Jupyter Notebook in VS Code**:
   - In VS Code, open the Jupyter Notebook file that you wish to run.
   - Click "Select Kernel" in the top-right corner of the notebook interface.
   - Select the kernel associated with your virtual environment at the path `.venv/bin/python`.
     - Note: In EAP you will select "Python Environments" then choose the starred environment at the path `.venv/bin/python`.
     - For local Python users, if you don't see your kernel immediately then go to `Select Another Kernel` > `Python Environment`, and you should find it listed there. If you do not see the kernel, restart VS Code.