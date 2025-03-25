from turtle import st

from sklearn.pipeline import Pipeline

def extract_inner_model(model):
    from pycaret.internal.pipeline import Pipeline
    if isinstance(model, Pipeline):
        return model.steps[-1][1]
    return model


