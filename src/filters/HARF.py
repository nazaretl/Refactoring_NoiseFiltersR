"""High Agreement Random Forest (HARF)

Ensemble-based filter for removing label noise from a dataset as a
preprocessing step of classification. Does not reclassify data.

NOTE: Class names in classes get truncated to their first character, can cause issues

TODO: Formalize docstrings

"""
from Filter import Filter
import pandas as pd
import sklearn.ensemble as skensemble
import dataclasses


@dataclasses.dataclass(frozen=False)
class HARF(Filter):

    """
    TODO: Formalize docstrings and explanations

    Parameters
    ----------

    data : pandas.DataFrame
        The two-dimensional numerical data to be filtered, organized such that each column is
        a property and each row is a datapoint. The last column should contain the classification of
        each datapoint.



    """

    data: pd.DataFrame | None = None
    nfolds: int = 10
    agreementLevel: float = 0.7
    ntrees: int = 500

    X: pd.DataFrame = dataclasses.field(init=False)
    y: pd.DataFrame = dataclasses.field(init=False)

    def __post_init__(self):
        if self.data is None:
            raise ValueError("The input data must be defined, is currently None")

        if self.data.ndim != 2:
            raise AttributeError(
                f"Data must be two-dimensional, currently {self.data.ndim}-dimensional"
            )
        if (self.agreementLevel < 0.5) | (self.agreementLevel > 1):
            raise ValueError(
                f"The agreement level of {self.agreementLevel} must be between 0.5 and 1"
            )

    def __apply_filter__(self):
        rand_forest = skensemble.RandomForestClassifier(
            n_estimators=self.ntrees, max_samples=self.nfolds, n_jobs=-1
        )

        rand_forest.fit(self.X, self.y)
        prob = pd.DataFrame(rand_forest.predict_proba(self.X))

        # Find the elements which do fit within the agreement limit
        is_clean = pd.Series(len(self.data), dtype=bool)
        for i in range(len(self.data)):
            is_clean[i] = (prob.iloc[i, :] > self.agreementLevel).any()
        return is_clean
