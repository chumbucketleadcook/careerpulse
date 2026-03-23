def run_cv(pipeline, X_tr, y_tr):
    """Returns mean/std accuracy and macro-F1 across CV folds."""
    results = cross_validate(
        pipeline, X_tr, y_tr,
        cv=cv,
        scoring=["accuracy", "f1_macro"],
        return_train_score=False,
        n_jobs=-1
        )
    return {
        "cv_accuracy_mean" : float(np.mean(results["test_accuracy"])),
        "cv_accuracy_std" : float(np.std(results["test_accuracy"])),
        "cv_f1_macro_mean" : float(np.mean(results["test_f1_macro"])),
        "cv_f1_macro_std" : float(np.std(results["test_f1_macro"])),
        }

def eval_on_test(pipeline, X_tr, y_tr, X_te, y_te):
    """Fits pipeline on full train set, evaluates on held-out test set."""
    pipeline.fit(X_tr, y_tr)
    preds = pipeline.predict(X_te)
    return {
        "test_accuracy" : float(accuracy_score(y_te, preds)),
        "test_f1_macro" : float(f1_score(y_te, preds, average="macro", zero_division=0)),
        }, preds
    
def run_knn_experiment(
    estimator,
    X_train,
    X_test,
    y_train,
    y_test,
    cv,
    params,
    results_list,
    run_name,
):
    """
    Runs CV and test set evaluation for a single KNN configuration,
    logs everything to MLflow, and appends results to results_list.

    Args:
        estimator    : fitted or unfitted sklearn estimator / pipeline
        X_train      : training features (dense array or sparse matrix)
        X_test       : test features
        y_train      : training labels
        y_test       : test labels
        cv           : sklearn CV splitter (e.g. StratifiedKFold)
        params       : dict of params to log to MLflow
        results_list : list to append the result row to
        run_name     : MLflow run name string
    """
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)

        # Cross-validation
        cv_results = cross_validate(
            estimator, X_train, y_train,
            cv=cv,
            scoring=["accuracy", "f1_macro"],
            return_train_score=False,
            n_jobs=-1
        )
        cv_metrics = {
            "cv_accuracy_mean" : float(np.mean(cv_results["test_accuracy"])),
            "cv_accuracy_std"  : float(np.std(cv_results["test_accuracy"])),
            "cv_f1_macro_mean" : float(np.mean(cv_results["test_f1_macro"])),
            "cv_f1_macro_std"  : float(np.std(cv_results["test_f1_macro"])),
        }
        mlflow.log_metrics(cv_metrics)

        # Test set evaluation
        estimator.fit(X_train, y_train)
        preds = estimator.predict(X_test)
        test_metrics = {
            "test_accuracy" : float(accuracy_score(y_test, preds)),
            "test_f1_macro" : float(f1_score(y_test, preds, average="macro", zero_division=0)),
        }
        mlflow.log_metrics(test_metrics)

        report = classification_report(y_test, preds, zero_division=0)
        mlflow.log_text(report, artifact_file="classification_report.txt")

        row = {**params, **cv_metrics, **test_metrics}
        results_list.append(row)

        print(
            f"[{run_name}] "
            f"cv_f1={cv_metrics['cv_f1_macro_mean']:.4f} "
            f"(±{cv_metrics['cv_f1_macro_std']:.4f})  "
            f"test_f1={test_metrics['test_f1_macro']:.4f}"
        )