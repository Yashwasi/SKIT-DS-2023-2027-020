SKIT-DS-2023-2027-020
# Rain Drop Volume Prediction Using AI
===

# 

# \## Project Overview

# 

# Rain Drop Volume Prediction Using AI is a project focused on analyzing raindrop data and developing a machine learning model to predict raindrop volume.

# 

# The project uses data analysis, preprocessing, visualization, and machine learning techniques to explore relationships between raindrop characteristics and volume.

# 

# \## Objectives

# 

# \- Collect and prepare raindrop datasets.

# \- Explore and understand the dataset.

# \- Identify missing values and duplicate records.

# \- Perform data preprocessing and feature engineering.

# \- Develop and evaluate machine learning models for raindrop volume prediction.

# \- Visualize dataset insights and model results.

# 

# \## Technologies Used

# 

# \- \*\*Programming Language:\*\* Python

# \- \*\*Data Analysis:\*\* Pandas, NumPy

# \- \*\*Data Visualization:\*\* Matplotlib, Plotly

# \- \*\*Machine Learning:\*\* Scikit-learn, XGBoost

# \- \*\*Deep Learning:\*\* PyTorch

# \- \*\*Version Control:\*\* Git and GitHub

# 

# \
## Dataset

The project uses raindrop measurement data containing the following fields:

- `DateTime`
- `Diameter_mm`
- `Velocity_m_s`
- `Volume_mm3`
- `Condition`
- `SourceFile`

The original master dataset contains 13,830,190 rows from 17 source files.

### Data Cleaning

Exact duplicate rows were identified by comparing all six columns.

- Original rows: 13,830,190
- Exact duplicate occurrences removed: 1,368,866
- Rows after deduplication: 12,461,324
- Remaining exact duplicate rows: 0

The cleaned dataset is saved locally as:

`data/raindrop_volume_deduplicated.csv`

The original dataset is preserved. Large dataset files are not stored in this GitHub repository. Obtain the dataset from the project team and update local file paths in scripts if necessary.

# \## Project Structure

# 

# ```text

# SKIT-DS-2023-2027-020/

# ├── data/

# ├── models/

# ├── notebooks/

# ├── src/

# │   └── clean\_data.py

# ├── check\_duplicates.py

# ├── count\_source\_rows.py

# ├── generate\_report.py

# ├── verify\_master\_data.py

# ├── requirements.txt

# ├── .gitignore

# └── README.md

# ```

# 

# \## Setup Instructions

# 

# \### 1. Clone the repository

# 

# ```bash

# git clone https://github.com/Yashwasi/SKIT-DS-2023-2027-020.git

# cd SKIT-DS-2023-2027-020

# ```

# 

# \### 2. Create and activate a virtual environment

# 

# Windows PowerShell:

# 

# ```powershell

# python -m venv .venv

# .\\.venv\\Scripts\\Activate.ps1

# ```

# 

# \### 3. Install dependencies

# 

# ```bash

# pip install -r requirements.txt

# ```

# 

# \### 4. Configure the dataset

# 

# Place the master ZIP dataset in the appropriate location and update `ZIP\_PATH` or `zip\_path` in the relevant Python scripts to match your computer.

# 

# \## Available Scripts

# 

# | Script | Purpose |

# |---|---|

# | `count\_source\_rows.py` | Counts total rows and rows per source file. |

# | `verify\_master\_data.py` | Checks row counts, missing values, source files, and duplicate candidates. |

# | `check\_duplicates.py` | Estimates duplicate candidates using row hashes. |

# | `src/clean\_data.py` | Creates a working CSV copy from the ZIP dataset without modifying the original archive. |

# | `generate\_report.py` | Generates a PDF report using available Git history. |

# 

# Run a script from the repository root, for example:

# 

# ```powershell

# python count\_source\_rows.py

# ```

# 

# \*\*Note:\*\* `clean\_data.py` currently copies the dataset; it does not remove duplicates or perform actual data cleaning. Duplicate candidates should be investigated before removing any records.

# 

# \## Team Contributions

# 

# The project involves dataset preparation, data analysis, machine learning, backend development, dashboard development, and testing.

# 

# Individual responsibilities are assigned among team members according to the project plan.

# 

# \## Future Scope

# 

# \- Improve data preprocessing and feature engineering.

# \- Compare different machine learning models.

# \- Improve model evaluation and visualization.

# \- Explore rainfall intensity prediction as a possible future extension.

# 

# \## Repository

# 

# GitHub: https://github.com/Yashwasi/SKIT-DS-2023-2027-020



