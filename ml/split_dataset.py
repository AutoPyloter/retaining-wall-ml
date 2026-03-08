import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_FILE  = "data.csv"
RANDOM_SEED = 42

# Split ratios: 70% train / 20% test / 10% unseen
TRAIN_RATIO  = 0.70
TEST_RATIO   = 0.20
UNSEEN_RATIO = 0.10


def split_dataset(input_file=INPUT_FILE, random_seed=RANDOM_SEED):
    """
    Split the dataset into train, test, and unseen subsets.

    Ratios
    ------
    Train  : 70%
    Test   : 20%
    Unseen : 10%  (held out entirely — never seen during training or tuning)

    The unseen split is kept strictly separate to provide an unbiased
    evaluation of the final deployed model.

    Parameters
    ----------
    input_file  : str — Path to the full dataset CSV (semicolon-separated, comma decimal)
    random_seed : int — Random seed for reproducibility

    Output files
    ------------
    train.csv, test.csv, unseen.csv  (same format as input)
    """
    df = pd.read_csv(input_file, sep=";", decimal=",", header=0)

    # Step 1: Split into train (70%) and temporary remainder (30%)
    train_df, temp_df = train_test_split(df, test_size=1 - TRAIN_RATIO, random_state=random_seed)

    # Step 2: Split remainder into test (20%) and unseen (10%)
    # unseen fraction of temp = UNSEEN_RATIO / (TEST_RATIO + UNSEEN_RATIO) = 0.10/0.30 = 1/3
    unseen_fraction = UNSEEN_RATIO / (TEST_RATIO + UNSEEN_RATIO)
    test_df, unseen_df = train_test_split(temp_df, test_size=unseen_fraction, random_state=random_seed)

    # Save splits
    train_df.to_csv("train.csv",   sep=";", decimal=",", index=False)
    test_df.to_csv("test.csv",     sep=";", decimal=",", index=False)
    unseen_df.to_csv("unseen.csv", sep=";", decimal=",", index=False)

    print(f"Dataset split complete:")
    print(f"  Train  : {len(train_df):>5} rows ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Test   : {len(test_df):>5} rows ({len(test_df)/len(df)*100:.1f}%)")
    print(f"  Unseen : {len(unseen_df):>5} rows ({len(unseen_df)/len(df)*100:.1f}%)")


if __name__ == "__main__":
    split_dataset()
