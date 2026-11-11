scoreline_v1

Load with:
    import joblib
    lr = joblib.load('lr_pipeline.joblib')
    boost = joblib.load('boosting_model.joblib')

Inputs must have the exact feature columns listed in metadata.json, in that order.
Final probabilities = average of Elo, Poisson, logistic regression and boosting.

SECURITY: .joblib files can run code when loaded. Only load files that the team produced
and stored in this repository or a trusted location. Never load one from an unknown source.
