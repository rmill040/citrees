"""Conformal prediction for citrees.

Provides prediction sets (classification) and prediction intervals (regression)
with guaranteed coverage under exchangeability.

Split Conformal Prediction:
- Split data into training and calibration sets
- Train model on training set
- Compute nonconformity scores on calibration set
- Use scores to construct prediction regions with coverage guarantee

References:
- Vovk et al. (2005) - "Algorithmic Learning in a Random World"
- Angelopoulos & Bates (2021) - "A Gentle Introduction to Conformal Prediction"
- Lei et al. (2018) - "Distribution-Free Predictive Inference for Regression"
"""


import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin, clone
from sklearn.model_selection import train_test_split


class ConformalClassifier:
    """Conformal prediction wrapper for classifiers.

    Produces prediction sets with guaranteed coverage. A prediction set
    contains all classes whose probability exceeds a calibrated threshold.

    Uses Adaptive Prediction Sets (APS) method which produces smaller
    prediction sets than naive thresholding.

    Parameters
    ----------
    estimator : ClassifierMixin
        Base classifier with predict_proba method.
    alpha : float, default=0.1
        Target miscoverage rate. Prediction sets have 1-alpha coverage.
    calibration_size : float, default=0.2
        Fraction of data to use for calibration.
    random_state : int, optional
        Random seed for calibration split.

    Attributes
    ----------
    estimator_ : ClassifierMixin
        Fitted base estimator.
    qhat_ : float
        Calibrated quantile threshold.
    classes_ : np.ndarray
        Class labels.

    Examples
    --------
    >>> from citrees import ConditionalInferenceForestClassifier
    >>> from citrees._conformal import ConformalClassifier
    >>> base_clf = ConditionalInferenceForestClassifier(n_estimators=100)
    >>> clf = ConformalClassifier(base_clf, alpha=0.1)
    >>> clf.fit(X_train, y_train)
    >>> prediction_sets = clf.predict_set(X_test)  # List of class sets
    >>> # Each prediction set is guaranteed to contain the true class
    >>> # with probability >= 1 - alpha = 0.9
    """

    def __init__(
        self,
        estimator: ClassifierMixin,
        alpha: float = 0.1,
        calibration_size: float = 0.2,
        random_state: int | None = None,
    ):
        if not hasattr(estimator, "predict_proba"):
            raise ValueError("Estimator must have predict_proba method")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if not 0 < calibration_size < 1:
            raise ValueError("calibration_size must be in (0, 1)")

        self.estimator = estimator
        self.alpha = alpha
        self.calibration_size = calibration_size
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ConformalClassifier":
        """Fit the conformal classifier.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Training labels.

        Returns
        -------
        self
            Fitted conformal classifier.
        """
        # Split into training and calibration
        X_train, X_cal, y_train, y_cal = train_test_split(
            X, y, test_size=self.calibration_size, random_state=self.random_state, stratify=y
        )

        # Fit base estimator on training data
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X_train, y_train)
        self.classes_ = self.estimator_.classes_

        # Compute nonconformity scores on calibration data
        # Using APS: score = 1 - (cumsum of probs up to true class)
        proba_cal = self.estimator_.predict_proba(X_cal)
        scores = self._compute_scores(proba_cal, y_cal)

        # Compute (1-alpha) quantile with finite-sample correction
        n_cal = len(y_cal)
        q_level = np.ceil((n_cal + 1) * (1 - self.alpha)) / n_cal
        self.qhat_ = np.quantile(scores, min(q_level, 1.0))

        return self

    def _compute_scores(self, proba: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute APS nonconformity scores.

        Score = cumulative probability mass including true class, when classes are sorted by
        decreasing probability.
        """
        n = len(y)
        scores = np.zeros(n)

        for i in range(n):
            # Sort probabilities in decreasing order
            sorted_idx = np.argsort(-proba[i])
            cumsum = 0.0

            # Find cumulative sum including true class
            for idx in sorted_idx:
                cumsum += proba[i, idx]
                if self.classes_[idx] == y[i]:
                    # Add randomization for exact coverage
                    u = np.random.uniform(0, proba[i, idx])
                    scores[i] = cumsum - u
                    break

        return scores

    def predict_set(self, X: np.ndarray) -> list[set]:
        """Predict conformal prediction sets.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features.

        Returns
        -------
        list[set]
            List of prediction sets, one per sample. Each set contains
            the classes in the prediction region.
        """
        proba = self.estimator_.predict_proba(X)
        n = len(X)
        prediction_sets = []

        for i in range(n):
            # Sort probabilities in decreasing order
            sorted_idx = np.argsort(-proba[i])
            cumsum = 0.0
            pred_set = set()

            # Add classes until cumsum exceeds qhat
            for idx in sorted_idx:
                cumsum += proba[i, idx]
                pred_set.add(self.classes_[idx])
                if cumsum >= self.qhat_:
                    break

            prediction_sets.append(pred_set)

        return prediction_sets

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels (point predictions).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted class labels.
        """
        return self.estimator_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features.

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
            Predicted class probabilities.
        """
        return self.estimator_.predict_proba(X)

    def coverage(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute empirical coverage on test data.

        Parameters
        ----------
        X : np.ndarray
            Test features.
        y : np.ndarray
            True labels.

        Returns
        -------
        float
            Fraction of samples where true label is in prediction set.
        """
        pred_sets = self.predict_set(X)
        covered = sum(1 for i, ps in enumerate(pred_sets) if y[i] in ps)
        return covered / len(y)

    def average_set_size(self, X: np.ndarray) -> float:
        """Compute average prediction set size.

        Parameters
        ----------
        X : np.ndarray
            Test features.

        Returns
        -------
        float
            Average number of classes in prediction sets.
        """
        pred_sets = self.predict_set(X)
        return np.mean([len(ps) for ps in pred_sets])


class ConformalRegressor:
    """Conformal prediction wrapper for regressors.

    Produces prediction intervals with guaranteed coverage using
    split conformal prediction.

    Parameters
    ----------
    estimator : RegressorMixin
        Base regressor.
    alpha : float, default=0.1
        Target miscoverage rate. Intervals have 1-alpha coverage.
    calibration_size : float, default=0.2
        Fraction of data to use for calibration.
    symmetric : bool, default=True
        If True, produces symmetric intervals [y_hat - q, y_hat + q].
        If False, produces asymmetric intervals using quantile regression
        (requires estimator to support quantile predictions).
    random_state : int, optional
        Random seed for calibration split.

    Attributes
    ----------
    estimator_ : RegressorMixin
        Fitted base estimator.
    qhat_ : float
        Calibrated quantile for interval width.

    Examples
    --------
    >>> from citrees import ConditionalInferenceForestRegressor
    >>> from citrees._conformal import ConformalRegressor
    >>> base_reg = ConditionalInferenceForestRegressor(n_estimators=100)
    >>> reg = ConformalRegressor(base_reg, alpha=0.1)
    >>> reg.fit(X_train, y_train)
    >>> lower, upper = reg.predict_interval(X_test)
    >>> # The interval [lower[i], upper[i]] contains y_test[i]
    >>> # with probability >= 1 - alpha = 0.9
    """

    def __init__(
        self,
        estimator: RegressorMixin,
        alpha: float = 0.1,
        calibration_size: float = 0.2,
        symmetric: bool = True,
        random_state: int | None = None,
    ):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if not 0 < calibration_size < 1:
            raise ValueError("calibration_size must be in (0, 1)")

        self.estimator = estimator
        self.alpha = alpha
        self.calibration_size = calibration_size
        self.symmetric = symmetric
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ConformalRegressor":
        """Fit the conformal regressor.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Training targets.

        Returns
        -------
        self
            Fitted conformal regressor.
        """
        # Split into training and calibration
        X_train, X_cal, y_train, y_cal = train_test_split(
            X, y, test_size=self.calibration_size, random_state=self.random_state
        )

        # Fit base estimator on training data
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X_train, y_train)

        # Compute residuals on calibration data
        y_cal_pred = self.estimator_.predict(X_cal)
        residuals = np.abs(y_cal - y_cal_pred)

        # Compute (1-alpha) quantile with finite-sample correction
        n_cal = len(y_cal)
        q_level = np.ceil((n_cal + 1) * (1 - self.alpha)) / n_cal
        self.qhat_ = np.quantile(residuals, min(q_level, 1.0))

        return self

    def predict_interval(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict conformal prediction intervals.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features.

        Returns
        -------
        lower : np.ndarray of shape (n_samples,)
            Lower bounds of prediction intervals.
        upper : np.ndarray of shape (n_samples,)
            Upper bounds of prediction intervals.
        """
        y_pred = self.estimator_.predict(X)
        lower = y_pred - self.qhat_
        upper = y_pred + self.qhat_

        return lower, upper

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values (point predictions).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted target values.
        """
        return self.estimator_.predict(X)

    def coverage(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute empirical coverage on test data.

        Parameters
        ----------
        X : np.ndarray
            Test features.
        y : np.ndarray
            True targets.

        Returns
        -------
        float
            Fraction of samples where true value is in prediction interval.
        """
        lower, upper = self.predict_interval(X)
        covered = np.sum((y >= lower) & (y <= upper))
        return covered / len(y)

    def average_interval_width(self, X: np.ndarray) -> float:
        """Compute average prediction interval width.

        Parameters
        ----------
        X : np.ndarray
            Test features.

        Returns
        -------
        float
            Average interval width.
        """
        lower, upper = self.predict_interval(X)
        return np.mean(upper - lower)


class CQR(ConformalRegressor):
    """Conformalized Quantile Regression.

    Produces prediction intervals that adapt to local uncertainty
    by using quantile regression estimates. Requires a base estimator
    that can produce quantile predictions.

    For citrees, we approximate quantile predictions using the
    distribution of predictions from individual trees in a forest.

    Parameters
    ----------
    estimator : RegressorMixin
        Base regressor (should be a forest with estimators_ attribute).
    alpha : float, default=0.1
        Target miscoverage rate.
    calibration_size : float, default=0.2
        Fraction of data for calibration.
    random_state : int, optional
        Random seed.

    References
    ----------
    Romano et al. (2019) - "Conformalized Quantile Regression"
    """

    def __init__(
        self,
        estimator: RegressorMixin,
        alpha: float = 0.1,
        calibration_size: float = 0.2,
        random_state: int | None = None,
    ):
        super().__init__(
            estimator=estimator,
            alpha=alpha,
            calibration_size=calibration_size,
            symmetric=False,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CQR":
        """Fit CQR.

        Parameters
        ----------
        X : np.ndarray
            Training features.
        y : np.ndarray
            Training targets.

        Returns
        -------
        self
            Fitted CQR regressor.
        """
        # Split into training and calibration
        X_train, X_cal, y_train, y_cal = train_test_split(
            X, y, test_size=self.calibration_size, random_state=self.random_state
        )

        # Fit base estimator
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X_train, y_train)

        # Get quantile predictions on calibration set
        lower_quantile, upper_quantile = self._predict_quantiles(X_cal)

        # Compute conformity scores: max of lower and upper violations
        scores = np.maximum(lower_quantile - y_cal, y_cal - upper_quantile)

        # Compute quantile with finite-sample correction
        n_cal = len(y_cal)
        q_level = np.ceil((n_cal + 1) * (1 - self.alpha)) / n_cal
        self.qhat_ = np.quantile(scores, min(q_level, 1.0))

        return self

    def _predict_quantiles(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict lower and upper quantiles.

        For forests, uses the distribution of tree predictions. For single trees, falls back to
        point prediction ± 0.
        """
        if hasattr(self.estimator_, "estimators_"):
            # Forest: use tree predictions
            n = len(X)
            tree_preds = np.array([tree.predict(X) for tree in self.estimator_.estimators_])

            lower = np.percentile(tree_preds, self.alpha / 2 * 100, axis=0)
            upper = np.percentile(tree_preds, (1 - self.alpha / 2) * 100, axis=0)
        else:
            # Single tree: no distributional information
            pred = self.estimator_.predict(X)
            lower = pred
            upper = pred

        return lower, upper

    def predict_interval(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict CQR intervals.

        Parameters
        ----------
        X : np.ndarray
            Test features.

        Returns
        -------
        lower : np.ndarray
            Lower bounds.
        upper : np.ndarray
            Upper bounds.
        """
        lower_q, upper_q = self._predict_quantiles(X)
        lower = lower_q - self.qhat_
        upper = upper_q + self.qhat_

        return lower, upper
