# schneider-electric-deforestation

This Repo is created for the Zero deforestation mission hackathon. An AlexNet model was trained on the training set provided using Pytorch

# Setup
```
pyenv install 3.9.1 #install python 3.9.1

pipenv shell && pipenv install #initiate vitual enviroment 
```

# Run pipeline
- First you need to download and save the training and test data in the root directory as follows:
.
├── ...
├── data
│   ├── train_test_data          # Load and stress tests
│   ├── train.csv                # End-to-end, integration tests (alternatively `e2e`)
│   └── test.csv                 # Unit tests
└── ...

- you can then run the pipeline using `dvc repro`. the pipeline stages, outpots and dependencies can be found in `dvc.yaml`

```
dvc repro
```