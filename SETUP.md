# Local Environment Setup Guide

This guide details the step-by-step instructions for setting up the local development environment for this project on **Windows 10 (64-bit)**.

---

## 📋 System Prerequisites

Ensure you have the following software installed before proceeding:

* **Operating System**: Windows 10 (64-bit / `x64`).
* **Visual Studio 2022**: Installed with the **Desktop development with C++** workload.
* **Python 3.12 (64-bit)**: Installed with the `Add python.exe to PATH` option enabled during setup.
* **CMake (3.20+)**: Required for building third-party native C++ dependencies.
* **Visual Studio Code**: Installed with the **Python** and **Jupyter** extensions.

---

## 🐍 Step 1: Python Virtual Environment & Jupyter Setup

### 1.1 Create and Activate Virtual Environment
Open **PowerShell** or **Command Prompt** at the project root directory and execute:

```powershell
# Navigate to the project root directory
cd path\to\your\svm-hmm-approx

# Create a 64-bit Python 3.12 virtual environment
python -m venv venv

# Activate the virtual environment
# PowerShell:
.\venv\Scripts\Activate.ps1

# Command Prompt:
.\venv\Scripts\activate.bat
```

> **Note on PowerShell Execution Policy:**  
> If PowerShell prevents script execution with a `Restricted` error, run:  
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### 1.2 Install Required Packages
With the virtual environment activated, install the required packages for web scraping, data processing, and visualization:

```powershell
python -m pip install --upgrade pip
pip install pandas numpy matplotlib selenium webdriver-manager ipykernel
```

### 1.3 Configure Visual Studio Code
1. Open Visual Studio Code (`code .`).
2. Install the **Python** and **Jupyter** extensions if not already present.
3. Open any `.ipynb` file in the root directory.
4. Click **Select Kernel** in the upper-right corner of the editor.
5. Select the Python interpreter from your newly created virtual environment:  
   `.\venv\Scripts\python.exe`

---

## 🛠️ Step 2: Build C++ Dependencies (NLopt via CMake)

This project requires **NLopt** (a nonlinear optimization library). Follow these instructions to build the library binaries for 64-bit Windows. The rest of the libs are header-only and don't require any extra action.

The source code for NLOpt is already present in the repository under `estimation-engine/nlopt-2.11.0`

Open **Developer PowerShell for VS 2022** (or **x64 Native Tools Command Prompt for VS 2022**) and run:

```powershell
# Navigate to the NLopt directory
cd estimation-engine/nlopt-2.11.0

# Create and enter the build directory
mkdir build
cd build

# Configure CMake project for Visual Studio 2022 x64
cmake -G "Visual Studio 17 2022" -A x64 -DBUILD_SHARED_LIBS=ON ..

# Compile Release & Debug configurations
cmake --build . --config Release
cmake --build . --config Debug
```

After compilation, the target artifacts will be located at:
* Import Library: `external/nlopt/build/Release/nlopt.lib`
* Dynamic Link Library: `external/nlopt/build/Release/nlopt.dll`

---

## ⚙️ Step 3: Visual Studio Solution Configuration

### 3.1 Path & Macro Settings
The solution configuration uses relative macros to map project dependencies automatically. Verify the project properties in Visual Studio (`forward-algorithm.sln`):

* **Platform Target**: `x64`
* **C/C++ > General > Additional Include Directories**: Ensure the paths are configured correctly
* **Linker > General > Additional Library Directories**: Ensure the paths are configured correctly
* **Linker > Input > Additional Dependencies**: `nlopt.lib`

### 3.2 Header-Only Libraries
Header-only libraries included in the repository require no build step. Ensure their parent directories are added under **C/C++ > General > Additional Include Directories**.

### 3.3 Post-Build DLL Deployment Script
The Visual Studio solution contains a post-build event that automatically deploys `nlopt.dll` to the output executable directory to satisfy runtime dependencies.

If configuring manually, check **Project Properties > Build Events > Post-Build Event > Command Line**:
```cmd
xcopy /Y /D "$(SolutionDir)..\nlopt-2.11.0\build\Release" "$(OutDir)"
```

---

## 🧪 Step 4: Verification & Build Execution

1. Launch `forward-algorithm.sln` in **Visual Studio 2022**.
2. Set the build configuration to **Debug** and platform to **x64**.
3. Select **Build > Rebuild Solution** (`Ctrl+Shift+B`).
4. Press `F5` to execute. The output directory (`x64/Debug/`) will contain the executable alongside `nlopt.dll`.

---

## ❓ Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| `LNK2019: Unresolved External Symbol` | Architecture mismatch (e.g., linking x86 build against x64 app). | Rebuild NLopt explicitly specifying `-A x64` in CMake. |
| `nlopt.dll was not found` | The dynamic library was not copied to the target output directory. | Ensure the post-build script ran successfully, or manually copy `nlopt.dll` into `x64/Debug/`. |
| `WebDriverException` in Selenium | Browser version or driver mismatch. | Ensure `webdriver_manager` is used to instantiate the driver dynamically in Python. |
