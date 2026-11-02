"""
Module de pretraitement partage entre le notebook d'analyse et le script
de production. Contient la classe MarginLowImputer necessaire pour charger
le pipeline sauvegarde (model_faux_billets.joblib).
"""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression


class MarginLowImputer(BaseEstimator, TransformerMixin):
    """Impute les valeurs manquantes de margin_low par regression lineaire
    a partir des autres variables geometriques du billet."""

    COLUMNS = ["diagonal", "height_left", "height_right", "margin_low", "margin_up", "length"]

    def __init__(self):
        self.features_impute = ["diagonal", "height_left", "height_right", "margin_up", "length"]
        self.reg = LinearRegression()

    def fit(self, X, y=None):
        X = pd.DataFrame(X, columns=self.COLUMNS)
        train_complete = X.dropna(subset=["margin_low"])
        self.reg.fit(train_complete[self.features_impute], train_complete["margin_low"])
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.COLUMNS).copy()
        mask = X["margin_low"].isna()
        if mask.sum() > 0:
            X.loc[mask, "margin_low"] = self.reg.predict(X.loc[mask, self.features_impute])
        return X
