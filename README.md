
# RAF-AG: Report Analysis Framework for Automatic Attack Path Generation

This is a code repo for RAF-AG paper. RAF-AG is a framework suitable for generating attack paths for Cyber Threat Intelligence report.
Some usage of RAF-AG:
- 1. Generate attack paths for input CTI reports
- 2. The attack path follows the sequential order of information presented in CTI reports. While this is not true for all report, sequential order allows the understanding of causal relationship
- 3. The output attack path contains a list of MITRE ATT&CK's technique ID. Which is useful for future analysis of these reports


## Prerequisites
- The source code is written in Python language. A virtual env should be created to prevent conflicts with existing libraries in your system.
- Install library from requirements.txt. 
```bash
pip install -r /path/to/requirements.txt
```
- Download the pretrained language models for SpaCy to work. Both the language models (web_lg and web_tft ) are required to work.
```bash
python -m spacy download en_core_web_lg
python -m spacy download en_core_web_trf
```
- Install the Coreferee library to extract coreferences in text
```bash
python -m coreferee install en
```
## Quick Start
To start using the framework, simply put your CTI reports (each in a txt format) in the data/campaign/input folder. After that, simply run:
```bash
python3 main.py
```
When the analysis is done, look for the results in data/campain/decoding_resutl. We consider the name of the input CTI reports to be its ID. We will find the corresponding results for each report by looking for its name.

## Code Hierrachy
The RAF-AG is implemented using object oriented programming approach. There are main folders and classes:
- classes: 
    - sentence.py: This class is used to present a Sentence object. Each sentence will be analysze using the sentence dependency tree to extract the cybersecurity events
    - paragraph.py: This class is used to present a Paragraph object, containing a list of Sentences
    - procedure.py: This class is used to present a procedure example from MITRE ATT&CK framwork, inheriting from paragraph
    - campaign.py and bigcampaign.py: These classes are used to present a campaign, which is a CTI report.
    - preprocessings.py: This file contains functions for preprocessing the text
    - subject_verb_object_extraction: This file contains a list of Python functions, each present a grammar rules. These functions are used to traverse the sentence dependency tree to extract the relations.
    - consine_similarity.py: This file contains functions to extract the embeddings of phrases and calculate the cosine similarity between them.
- data: Data used in the framework. This is also the place for user to input new CTI reports for analyzing. The data hierrachy will be explained in the next section.
- keys.py: Hyperparameters of RAF-AG. This is where the users can customize the hyperparameters of the framework
- language_models.py: Prepare the lanauge_models before it can be used in RAF-AG
- mitre_attack.py: This script is used to read the ATT&CK STIX data and extract the require information (e.g, techniques, procedure examples)
- modules and utils: supporting modules and utils

## Data Hierrachy
There are different types of data folders you need to consider before you can run the code:
- data/campaign: 
    - input: where the input CTI reports are stored. To keep it simple, each CTI report will be saved in a single text file (*.txt). The name of the file is also the ID of the report
    - output: Where the graph data for the input CTI report is stored. Each CTI will be transformed into a graph and stored in jsonl format. 
    - procedure_alignment: This is the raw alignment results between the report graph and procedure graphs
    - decoding_result: This is the attack paths for CTI reports.
- data/procedure:
    - input: Where the raw data from ATT&CK is stored. We saved it in a single procedures.json file, containing all the procedure examples from ATT&CK. Main data for each procedure example contains: technique ID, platforms, and description.
    - output: Where the graph data for procedure examples are stored. In this folder, each procedure graph is stored as a single json file.
    - deduplication: Where the graph data for procedure examples are stored. However, this graph data is deduplicated, similar procedure graphs (within same technique) will be merge into one
    - analyzed_procedure.jsonl: All the data contained in data/procedure/deduplication are converted into a single jsonline file for faster import into the framework
- data/meta data: Some meta data used for different stages of RAF-AG. The most important is the verb_similarity data
- data/patterns: some generic regex patterns to work with ATT&CK-specific data. This is used for automatically extract keywords, special phrases from ATT&CK
- data/dictionarydata: This is the dictionary data for the weak supervision approach in RAF-AG. Each json file is used for one specific category (e.g., Actor.json)

## Acknowledgements
