
from datetime import datetime
from sklearn.model_selection import train_test_split

from core.mlflow_logger import get_experiment_logger
from core.session import session
import utils.preprocess_helper as pre
import utils.path_helper as ph
import utils.evaluation_helper as eval
import utils.mlflow_helper as mh
from D1_evaluation import evaluate_escalation


def single_run(X, y, model):
    exp_logger = get_experiment_logger()
    event_logger = exp_logger.logger

    random_state=session.parameters.get("random_state", 42)
    num_feats=session.?
    cat_feats=session.?

    (X_train, X_test,
    y_train, y_test) = train_test_split(
                                        X, y,
                                        test_size=0.2, 
                                        random_state=random_state, 
                                        stratify=y
                                        )

    # pre-training checks
    pre.pretraining_checks(X_train, X_test,y_train, y_test)
    # assert list(X_train.columns) == list(X_test.columns)

    # train model
    now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    event_logger.info("Start LogReg training: %s", now)
    session.now = now

    model.fit(X_train, y_train)

    # save coef_df and model + Logging
    eval.save_coef_df(num_feats, cat_feats, pipe=model)
    model_id = session.parameters.get("model_id", None)
    if model_id is None: 
        # model = pipe
        model_name = session.parameters.get("model_name")
        exp_logger.log_model(model_name, model, X_test)
        mh.save_model(model, model_name)

    # post-training check
    event_logger.info("Labels: %s", model.named_steps["clf"].classes_)

    y_pred = model.predict(X_test)
    # y_proba = pipe.predict_proba(X_test)[:, 1]

    # evaluate results
    best_t, thresh_df = eval.threshold_sweep_analysis(model, X_test, X_train, y_train, metric="f2")

    thresh_path = ph.create_save_path("thresh_df", "_thresh_df", ".csv")
    thresh_df.to_csv(thresh_path)

    # final eval on test (fixed threshold!)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_test_proba >= best_t).astype(int)

    evaluate_escalation(y_test, y_pred)
    eval.compile_roc_pr_auc(y_test, y_test_proba, data_viz=True)

    return 