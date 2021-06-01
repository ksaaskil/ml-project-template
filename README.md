# ML Project template

Example project template inspired by [this video](https://www.youtube.com/watch?v=ArygUBY0QXw) by the Kaggle Grandmaster Abhishek Thakur.

See also [`mlframework`](https://github.com/abhi1thakur/mlframework) for an example project.

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
$ python -m src.create_folds
```

### Train a model

```bash
$ python -m src.train
```

## Folder structure

- `src`: source code
- `input`: datasets
- `models`: model artifacts

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
