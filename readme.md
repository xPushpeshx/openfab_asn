# AI Junior Developer Test 
Welcome! You’ve stepped into the arena – now show us what you’ve got! 

## Mission
main.py file updated.

```bash
docker build . -t test
docker run -p5500:5500 test
```

## Explanation and code updated
I have used hugging face model for our science based QnA.
```python
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
```