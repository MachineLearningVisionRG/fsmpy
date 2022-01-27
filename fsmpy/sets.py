from __future__ import annotations
from collections.abc import Iterable
import numpy as np


class FuzzySet:
    """ Object to represent a Fuzzy Sets.

    Attributes
    ----------
    membership_values : 1-d np.ndarray
        Membership values of the set.
    non_membership_values : 1-d np.ndarray
        Non-membership values of the set.

    Notes
    --------
    An important note as to how the Fuzzy Sets are actually represented.
    For three patterns denoted in Fuzzy Sets in a universe 
    :math:`U=\{u_1, u_2, u_3\}`
    
    mathematically, they are represented as follows :math:`(μ, ν)`
    """
    def __init__(self, membership_values : Iterable, non_membership_values : Iterable = None):
        """ Constructor method.

        For single valued sets, use an Iterable with a single value.

        Parameters
        ----------
        membership_values : 1-d Iterable
            Membership values of the set.
        non_membership_values : 1-d Iterable, optional
            Non-membership values of the set. If not provided, an np.array is 
            initialized with zeros with the same size as membership_values.
        
        """
        self.membership_values = np.array(membership_values)
        if non_membership_values is None:
            self.non_membership_values = np.zeros_like(self.membership_values)
        else:
            self.non_membership_values = np.array(non_membership_values)

    def __setattr__(self, name : str, value : Iterable):
        """ Sets the values of the specified attribute.
        
        Parameters
        ----------
        name : str
            Name of the attribute to set.
        value : Iterable
            The values to set to the attribute.
        
        Raises
        ------
        TypeError if value is not an Iterable or a numpy array
        """
        if not isinstance(value, Iterable):
            raise TypeError(
                "{} must be an Iterable or a numpy array, not {}!".format(name, type(value)))

        self.__dict__[name] = np.array(value)

    def __len__(self):
        """ Returns cardinality of the set.
        
        Returns the size of the membership_values.

        Returns
        -------
        int
        """
        return len(self.membership_values)

    def __str__(self):
        """ String representation of the set.
        
        Returns
        -------
        str
        """
        return "μ: {},\nv:{}".format(self.membership_values, self.non_membership_values)


class IntuitionisticFuzzySet(FuzzySet):
    """ Object to represent a Intuitionistic Fuzzy Sets. 

    The important difference from FuzzySet is that it incorporates the
    hesitation degree values.

    Attributes
    ----------
    membership_values : 1-d np.ndarray
        Membership values of the set.
    non_membership_values : 1-d np.ndarray
        Non-membership values of the set.
    hesitation_degrees : 1-d np.ndarray
        Hesitation degree values of the set.

    Notes
    --------
    An important note as to how the Intuitionistic Fuzzy Sets are actually represented.
    For three patterns denoted in Intuitionistic Fuzzy Sets in a universe 
    :math:`U=\{u_1, u_2, u_3\}`
    
    mathematically, they are represented as follows :math:`(μ, ν)`
    """
    def __init__(self, membership_values: Iterable, non_membership_values: Iterable = None, hesitation_degrees: Iterable = None):
        """ Constructor method.

        For single valued sets, use an Iterable with a single value.

        Parameters
        ----------
        membership_values : 1-d Iterable
            Membership values of the set.
        non_membership_values : 1-d Iterable, optional
            Non-membership values of the set. If not provided, an np.array is 
            initialized with zeros with the same size as membership_values.
        hesitation_degrees : 1-d Iterable, optional
            Hesitation degrees values of the set. If not provided, it is set to

            >>> hesitation_degrees = 1.0 - membership_values - self.non_membership_values
        
        """
        super().__init__(membership_values, non_membership_values)
        if hesitation_degrees is None:
            self.hesitation_degrees = 1.0 - np.array(membership_values) - np.array(self.non_membership_values)
        else:
            self.hesitation_degrees = hesitation_degrees

    def __len__(self):
        """ Returns cardinality of the set.
        
        Returns the size of the membership_values.

        Returns
        -------
        int
        """
        return len(self.membership_values)

    def __str__(self):
        """ String representation of the set.
        
        Returns
        -------
        str
        """
        return "μ: {}\nv:{}\nπ: {}".format(self.membership_values, self.non_membership_values, self.hesitation_degrees)
