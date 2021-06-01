import pandas as pd
from sklearn import model_selection

if __name__ == '__main__':
    # Create folds for cross-validation
    df = pd.read_csv("input/train.csv")

    df["kfold"] = -1

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    kf = model_selection.StratifiedKFold(n_splits=5, shuffle=False)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X=df, y=df.target.values)):
        # print('fold', fold, len(train_idx), len(val_idx))
        df.loc[val_idx, 'kfold'] = fold

    output_csv = "input/train_folds.csv"

    print(f"Writing to: {output_csv}")
    df.to_csv(output_csv, index=False)
