from collections.abc import Callable
from typing import Iterable, List
import warnings
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
import numpy as np

from ..sets import IntuitionisticFuzzySet


def classify(class_patterns: Iterable[IntuitionisticFuzzySet], sample_pattern: IntuitionisticFuzzySet, 
            measure_caller: Callable, *, is_distance=True, return_confidence=False, **kwargs) -> np.intp:
    """ Simple classification method to classify a sample pattern given class patterns, using the measure provided.

    For each class pattern c, calculates the measure from measure_caller between c and sample_pattern.
    The class is chosen by the min/max measure between the sample and class patterns, depending on is_distance.
    If return_confidence is true, returns the degree of confidence for the chosen class.

    Parameters
    ----------
    class_patterns : list[IntuitionisticFuzzySet]
        Class patterns to which the sample_pattern is classified.
    sample_pattern : IntuitionisticFuzzySet
        The sample to be classified.
    measure_caller : Callable
        The measure function to use when measuring the sets of the ideally thresholded image and the current image set.
    is_distance : bool
        If the measure provided is a distance or a similarity. Used to pick the best measure calculated.
    return_confidence : bool
        Whether the confidence degree is returned. Can only be calculated if is_distance is True.
    **kwargs : additional arguments
        Passed to the measure_caller.
    
    Returns
    -------
    numpy.intp
        The class of sample_pattern.
    
    See Also
    --------
    fsmpy.utils.confidence_degree
    """
    if not is_distance and return_confidence:
        warnings.warn("Degree of Confidence can only be calculated for distances. Returning only prediction.")
        return_confidence = False
    measures = [
        measure_caller(sample_pattern, class_pattern, **kwargs) * (1 if is_distance else -1)
        for class_pattern in class_patterns
    ]
    prediction = np.argmin(measures)
    if return_confidence:
        from . import confidence_degree
        return prediction, confidence_degree(np.min(measures), measures[:prediction] + measures[prediction:])
    else:
        return prediction


class FuzzyTextClassifier(BaseEstimator, ClassifierMixin):
    """ Text document classification proposed by P. Intarapaiboon from the related article: 
    "Text classification using similarity measures on intuitionistic fuzzy sets".

    Train and classification object to classify text documents following
    the proposed method.

    Follows Estimator API from scikit-learn Estimator object.

    Attributes
    ----------
    measure_caller : Callable
        Measure to use during prediction process.
    measure_kwargs : dict
        Passed to the measure_caller.
    is_distance : bool
        If measure_caller is a distance or not.

    Methods
    -------
    fit : FuzzyTextClassifier
        "Trains" the classifier.
    predict : list[np.intp]
        List of predicted class for each sample
    predict_proba : list[list[np.float64]]
        List of the measure results between each class, for each sample.
    get_params : dict
        Parameters of the classifier.
    set_params : FuzzyTextClassifier
        Sets the parameters of the classifier.

    See also
    --------
    sklearn.base.BaseEstimator
    
    """
    def __init__(self, measure_caller: Callable, *, is_distance: bool = True, **measure_kwargs):
        """ Constructor method.
        
        Parameters
        ----------
        measure_caller : Callable
            The measure to be tested.
        is_distance : bool
            If the measure provided is a distance or a similarity. Used to pick the best measure calculated.
        **measure_kwargs : measure arguments
            Passed to the measure_caller.
        """
        self.measure_caller = measure_caller

        self._param_names = ["measure_caller"] + list(measure_kwargs.keys())
        self._measure_kwargs = measure_kwargs
        self.class_patterns = None
        self.is_distance = is_distance

    def fit(self, X: Iterable[IntuitionisticFuzzySet], y: Iterable) -> object:
        """ "Trains" on the X data prpovided.
        
        Calculates the membership and non-membership values of each word for each unique class in y.

        Parameters
        ----------
        X : Iterable[IntuitionisticFuzzySet]
            Data to train upon.
        y : Iterabale
            Target vector relative to X.
            
        Returns
        -------
        self: Object
            An instance of the estimator.

        See also
        --------
        fsmpy.utils.calculate_documents_membership
        """
        if not all(isinstance(x, IntuitionisticFuzzySet) for x in X):
            raise TypeError(
                "Expected X to be an Iterable of types IntuitionisticFuzzySet or IntuitionisticFuzzySet, got {}".format(type(X)))

        y = np.array(y)
        X = np.array(X)
        
        self.classes_ = unique_labels(y)

        self.class_patterns = {_class: None for _class in self.classes_}
        for _class in self.classes_:
            same_class_docs = y == _class
            
            mean_m = np.average(
                [x.membership_values for x in X[same_class_docs]], axis=0)
            mean_v = np.average(
                [x.non_membership_values for x in X[same_class_docs]], axis=0)
            if hasattr(X[0], "hesitation_degrees"):
                mean_h = np.average(
                    [x.hesitation_degrees for x in X[same_class_docs]], axis=0)
                self.class_patterns[_class] = IntuitionisticFuzzySet(
                    mean_m, mean_v, mean_h)
            else:
                self.class_patterns[_class] = IntuitionisticFuzzySet(mean_m, mean_v)
        return self

    def predict(self, X: Iterable[IntuitionisticFuzzySet]) -> np.ndarray:
        """ Predict class labels for samples X.
        
        Calculates the membership and non-membership values of each word for each unity class in y.

        Parameters
        ----------
        X : Iterable[IntuitionisticFuzzySet]
            Samples.
            
        Returns
        -------
        list[np.intp]
            Predicted class label per sample.
        """
        if not all(isinstance(x, IntuitionisticFuzzySet) for x in X):
            raise TypeError(
                "Expected X to be an Iterable of types IntuitionisticFuzzySet or IntuitionisticFuzzySet, got {}".format(type(X)))

        predictions = []
        for x in X:
            predictions.append(
                np.argmin(
                    [
                        self.measure_caller(x, self.class_patterns[_class], **self._measure_kwargs) * (1 if self.is_distance else -1)
                        for _class in self.classes_
                    ]
                )
            )
        return np.array([self.classes_[pred] for pred in predictions])

    def predict_proba(self, X: Iterable[IntuitionisticFuzzySet]) -> np.ndarray:
        """ Measures of each sample.
        
        The returned values are the results returned when the measure_caller is called
        for all classes for each label.

        Parameters
        ----------
        X : Iterable[IntuitionisticFuzzySet]
            Samples.
            
        Returns
        -------
        list[list[float]]
            Returns the measures of the sample for each class, where classes are in self.classes_
        """
        if not all(isinstance(x, IntuitionisticFuzzySet) for x in X):
            raise TypeError(
                "Expected X to be an Iterable of types IntuitionisticFuzzySet or IntuitionisticFuzzySet, got {}".format(type(X)))

        probas = []
        for x in X:
            probas.append(
                [self.measure_caller(x, self.class_patterns[_class], **self._measure_kwargs)
                 for _class in self.classes_]
            )
        return np.vstack(probas)

    def get_params(self, deep=True) -> dict:
        params = {k: v for k, v in self._measure_kwargs.items()}
        params["measure_caller"] = self.measure_caller
        params["is_distance"] = self.is_distance
        return params

    def set_params(self, **parameters) -> object:
        for parameter, value in parameters.items():
            if parameter in ["is_distance", "measure_caller"]:
                setattr(self, parameter, value)
            else:
                self._measure_kwargs[parameter] = value
        return self
