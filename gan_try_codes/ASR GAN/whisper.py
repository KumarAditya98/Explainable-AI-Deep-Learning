import pandas as pd
from transformers import WhisperForConditionalGeneration
import os

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")

DATA_DIR = os.getcwd() + os.path.sep + 'en' + os.path.sep
os.chdir(DATA_DIR)

import csv

# Open the TSV file in read mode
with open('other.tsv', "r", encoding="utf-8") as f:

    # Create a CSV reader object
    reader = csv.reader(f, delimiter="\t")

    # Iterate over the rows in the CSV reader object
    for row in reader:

        # Process each row of the TSV file
        print(row)

    # Close the TSV file
    f.close()

