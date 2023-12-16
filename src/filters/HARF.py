"""High Agreement Random Forest (HARF)

Ensemble-based filter for removing label noise from a dataset as a
preprocessing step of classification. Does not reclassify data.

"""
from Filter import Filter
import pandas as pd
import sklearn.ensemble as skensemble
import dataclasses


@dataclasses.dataclass(frozen=False)
class HARF(Filter):

    """
    Ensemble-based filter for removing label noise from a dataset as a
    preprocessing step of classification. Does not reclassify data.

    Attributes
    ----------

    data: pandas.DataFrame
        The two-dimensional numerical data to be filtered, organized such that each column is
        a property and each row is a datapoint. The last column should contain the classification of
        each datapoint.

    nfolds: int, default=10
        The number of segments to subdivide the input datapoints in the random forest training.

    agreementLevel: float, default=0.7
        The probability cutoff below which a datapoint is classified as noise if the random forest fitting does not 
        indicate a probabalistic classification that is likely enough.

    ntrees: int, default=500
        The number of trees in the random forest.

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
        is_clean = []
        for i in range(len(self.data)):
            is_clean.append((prob.iloc[i, :] > self.agreementLevel).any())
        return is_clean
