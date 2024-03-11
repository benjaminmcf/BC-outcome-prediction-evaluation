# BC-outcome-prediction-evaluation

## For the paper "Evaluation of machine learning pipeline for blood culture outcome prediction on prospectively collected data in Western Australian emergency department"


-------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------

## Set up the environment

#### ** It is important that you are using python 3.10.5 **

- This is using git bash

```bash

python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

- This is using command prompt

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```


## Running the Code


- Using Git Bash

```bash

cd scripts
./run_scripts.sh
```

This will run the evaluation.py and statistics.py scripts and output the results to the output_files folder

- Using Command Prompt

```bash
cd scripts
python evaluation.py > output_files/evaluation.txt
python statistics.py > output_files/statistics.txt
```

## Running the Code Manually

Individual code files can be run manually as well using the following commands. This will not output the results to the output_files folder.

```bash
python evaluation.py
python statistics.py
```
