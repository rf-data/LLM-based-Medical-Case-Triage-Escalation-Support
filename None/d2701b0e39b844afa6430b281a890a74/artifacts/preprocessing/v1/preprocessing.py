def prepare_reports(f_path, 
                    col_to_drop):
    exp_logger = get_experiment_logger()
    event_logger = exp_logger.logger
    
    # loading df + dropping columns unnecessary
    df = pd.read_csv(f_path)
    df = df.drop(columns=col_to_drop)
    
    # split dataset
    y = df["expected_action"].map({"no_escalation": 0, "escalation": 1}).copy()
    X = df.drop(columns=["expected_action"]).copy()

    event_logger.info("Head of X:\n%s", X.head(5))
    event_logger.info("\nHead of y:\n%s", y.head(5))

    # # debug prints
    # dup_rows = X.duplicated().mean()
    # print("-"*25, " [START DEBUGGING] ", "-"*25)
    # print("Duplicate share:", dup_rows)

    # # optional: check duplicates with target consistency
    # dups = df[df.duplicated(subset=X.columns.tolist(), keep=False)]
    # print("Duplicated rows:", len(dups))
    # print(dups["expected_action"].value_counts())
    # print("-"*25, " [END DEBUGGING] ", "-"*25)

    X_train, X_test,y_train, y_test = train_test_split(
                                                X, y,
                                                test_size=0.2, 
                                                random_state=42, 
                                                stratify=y
                                                )

    # pre-training check
    assert list(X_train.columns) == list(X_test.columns)

    return X_train, X_test, y_train, y_test
