import os
import warnings
from typing import Dict

from openfabric_pysdk.utility import SchemaUtil

from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText

from openfabric_pysdk.context import Ray, State
from openfabric_pysdk.loader import ConfigClass

from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer=None
model=None
############################################################
# Callback function called on update config
############################################################
def config(configuration: Dict[str, ConfigClass], state: State):
    global tokenizer, model

    if tokenizer is None:
        try:
            # Initialize the question-answering pipeline with a pre-trained model
            tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
        except Exception as e:
            print(f"Error initializing tokenizer: {e}")
            # Optionally handle the error here
    if model is None:
        try:
            model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl")
        except Exception as e:
            print(f"Error initializing the model : {e}")
    # Additional configuration or setup can be added here if needed
    pass


############################################################
# Callback function called on each execution pass
############################################################
def execute(request: SimpleText, ray: Ray, state: State) -> SimpleText:
    output = []
    for text in request.text:
        # Ensure the NLP pipeline is initialized before processing the request
        if tokenizer and model is None:
            output.append("Error: Model and Tokenizer not initialized.")
        else:
            # Utilize the pre-trained model to find the answer to the science-related question
            input_text = text
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids

            outputs = model.generate(input_ids)

            output.append(tokenizer.decode(outputs[0]))

    return SchemaUtil.create(SimpleText(), dict(text=output))
