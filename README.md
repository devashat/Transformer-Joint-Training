# Joint Training with a Transformer Encoder

This project requires training a Transformer model to perform two tasks simultaneously. The first task is tagging Natural Language Utterances using BIO tags, and the second task is relation extraction when the input is NLUs.

## Provided files

- `train.csv`: The train data for the project
- `test.csv`: The test data for the project
- `joint.ipynb`: The notebook used to develop the model and code for the project
- `main.py`: The python file that contains the final code for the project. It contains an argparse mechanism that can be used to train and test the model. The `--train` flag trains the model on the data, and the `--test` flag generates predictions for the test data.
- `requirements.txt`: Contains the list of require packages to set up an adequate environment for the project.

## Setup

***I did add os.system commands for the first three commands, but keeping this here just in case.***

To run this code, make sure to do the following first in cli:

```
wget http://vectors.nlpl.eu/repository/20/6.zip

unzip /content/6.zip

unzip 6.zip -d wikipedia

pip install -r requirements.txt
```

## Sample Run Command

### Training

```
python3 main.py --train --data "train.csv" --save_model "./trained_model.pt"
```

### Testing

```
python3 main.py --test --data "test.csv" --model_path "./trained_model.pt" --output "./predictions.csv"
```

