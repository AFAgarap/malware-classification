Contribution Guidelines
===

This document consists of conventions to follow when contributing to this project.

### Coding Convention

[PEP 8 Python Coding Convention](https://www.python.org/dev/peps/pep-0008/) is to be strictly observed in this repository.
To easily adhere to the said convention, it is recommended to use [PyCharm IDE](https://www.jetbrains.com/pycharm/).

### Project Structure

The dataset is partitioned to 70% / 30% setting, training and testing dataset respectively. All of which are in NPY format.
Meanwhile, the scripts written for data reading, loading, standardization, binning, and conversion to NPY are to be located in `utils` subdirectory.
Lastly, the implementation of the GAN model is to be found in `model` subdirectory.
```buildoutcfg
malware-classification-gan
├── CONTRIBUTING.md
├── dataset
|      ├── z-score
|      |     ├── train
|      |     └── test
|      ├── linear
|      |     ├── train
|      |     └── test
|      └── non-linear
|            ├── train
|            └── test
├── LICENSE
├── model
├── README.md
└── utils
```

### Git Commit Guidelines

Use [semantic commit messages](seesparkbox.com/foundry/semantic_commit_messages):
```buildoutcfg
quality (improving the quality of the data (images/labels))
feat (new feature)
fix (bug fix)
docs (changes to documentation)
style (formatting, missing semi colons, etc; no code change)
refactor (refactoring production code)
test (adding missing tests, refactoring tests; no production code change)
chore (updating grunt tasks etc; no production code change)
```

#### Examples

Take note to use present tense of verb at the beginning. Do not include two tasks in a single commit, e.g. bug fix and feature in one commit.
```buildoutcfg
chore : add script for reading images into vector
fix : add a condition to break infinite loop
refactor : restructure script to follow OOP
style : convert tabs to spaces
style : follow pep8 coding convention
feat : use sklearn for standardization
docs : explain preprocessing procedure
```
