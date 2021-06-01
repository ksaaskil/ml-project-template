# ML Project template

## Instructions

Install dependencies:

```bash
$ pip install -r requirements-dev.txt
```

### Download data

Download data to `input/` folder:

```bash
$ download.sh
```

### Create folds

```bash
$ python src/create_folds.py
```

### Train a model

```bash
$ python src/train.py
```

## Tips

### Add `.gitignore`

```bash
$ wget https://raw.githubusercontent.com/github/gitignore/master/Python.gitignore -O .gitignore
```

### Run `code-server`

Install [code-server](https://github.com/cdr/code-server) to run VS Code on any remote server ([instructions](https://github.com/cdr/code-server/blob/main/docs/install.md#macos))

```bash
$ wget https://raw.githubusercontent.com/cdr/code-server/main/install.sh
$ bash install.sh
```

Run `code-server`:

```bash
$ code-server
```

### Sample data

Download `train.csv` and `test.csv` from [this competition](https://www.kaggle.com/c/cat-in-the-dat/overview).
