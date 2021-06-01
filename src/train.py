import pandas as pd
from sklearn import ensemble
from sklearn import metrics
from sklearn import preprocessing

TRAINING_DATA = "input/train_folds.csv"

FOLD = 0

FOLD_MAPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3],
}


def example_main():

    df = pd.read_csv(TRAINING_DATA)
    train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))]
    val_df = df[df.kfold == FOLD]

    y_train = train_df.target.values
    y_val = val_df.target.values

    train_df = train_df.drop(columns=["id", "target", "kfold"])

    val_df = val_df.drop(columns=["id", "target", "kfold"])

    # Ensure columns are in the same order
    val_df = val_df[train_df.columns]

    label_encoders = []

    for c in train_df.columns:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(train_df[c].values.tolist() + val_df[c].values.tolist())
        train_df.loc[:, c] = lbl.transform(train_df[c].values.tolist())
        val_df.loc[:, c] = lbl.transform(val_df[c].values.tolist())
        label_encoders.append((c, lbl))

    clf = ensemble.RandomForestClassifier(n_estimators=20, verbose=2)

    clf.fit(train_df, y_train)

    y_pred = clf.predict_proba(val_df)[:, 1]

    print(metrics.roc_auc_score(y_val, y_pred))


if __name__ == "__main__":
    example_main()
