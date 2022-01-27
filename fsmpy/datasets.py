from .sets import FuzzySet


def load_patients_diagnoses():
    """ Returns diagnoses and patients data for the medical diagnosis application.

    The data returned are consisted of FuzzySets representing the symptoms each
    patient has and the symptoms of each diagnosis.
    The symptoms include the following:
    **Temperature**, **Headache**, **Stomach paint**, **Cough** and **Chest paint**.

        
    Returns
    -------
    diagnoses : list[FuzzySet]
        List of FuzzySets representing each giadnosis: 
        **Viral fever**, **Malaria**, **Typhoid**, **Stomach problem** and **Chest problem**. 
    patients : list[FuzzySet]
        List of FuzzySets representing each patient's symptoms:
        Al, Bob, Joe and Ted.
    
    Examples
    --------
    Diagnoses:

    >>> viral_fever = FuzzySet([0.4, 0.3, 0.1, 0.4, 0.1], [0.0, 0.5, 0.7, 0.3, 0.7])
    >>> malaria = FuzzySet([0.7, 0.2, 0.0, 0.7, 0.1], [0.0, 0.6, 0.9, 0.0, 0.8])
    >>> typhoid = FuzzySet([0.3, 0.6, 0.2, 0.2, 0.1], [0.3, 0.1, 0.7, 0.6, 0.9])
    >>> stomach_problem = FuzzySet([0.1, 0.2, 0.8, 0.2, 0.2], [0.7, 0.4, 0.0, 0.7, 0.7])
    >>> chest_problem = FuzzySet([0.1, 0.0, 0.2, 0.2, 0.8], [0.8, 0.8, 0.8, 0.8, 0.1])

    Patients:
    
    >>> al = FuzzySet([0.8, 0.6, 0.2, 0.6, 0.1], [0.1, 0.1, 0.8, 0.1, 0.6])
    >>> bob = FuzzySet([0.0, 0.4, 0.6, 0.1, 0.1], [0.8, 0.4, 0.1, 0.7, 0.8])
    >>> joe = FuzzySet([0.8, 0.8, 0.0, 0.2, 0.0], [0.1, 0.1, 0.6, 0.7, 0.5])
    >>> ted = FuzzySet([0.6, 0.5, 0.3, 0.7, 0.3], [0.1, 0.4, 0.4, 0.2, 0.4])
    """
    # diagnosis = [temperature, headache, stomach pain, cough, chest paint]
    viral_fever = FuzzySet([0.4, 0.3, 0.1, 0.4, 0.1],
                           [0.0, 0.5, 0.7, 0.3, 0.7])
    malaria = FuzzySet([0.7, 0.2, 0.0, 0.7, 0.1], [0.0, 0.6, 0.9, 0.0, 0.8])
    typhoid = FuzzySet([0.3, 0.6, 0.2, 0.2, 0.1], [0.3, 0.1, 0.7, 0.6, 0.9])
    stomach_problem = FuzzySet([0.1, 0.2, 0.8, 0.2, 0.2], [
                               0.7, 0.4, 0.0, 0.7, 0.7])
    chest_problem = FuzzySet([0.1, 0.0, 0.2, 0.2, 0.8], [
                             0.8, 0.8, 0.8, 0.8, 0.1])
    diagnoses = [viral_fever, malaria, typhoid, stomach_problem, chest_problem]

    # patient = membership values for each characteristic, non-membership values for each characteristic
    al = FuzzySet([0.8, 0.6, 0.2, 0.6, 0.1], [0.1, 0.1, 0.8, 0.1, 0.6])
    bob = FuzzySet([0.0, 0.4, 0.6, 0.1, 0.1], [0.8, 0.4, 0.1, 0.7, 0.8])
    joe = FuzzySet([0.8, 0.8, 0.0, 0.2, 0.0], [0.1, 0.1, 0.6, 0.7, 0.5])
    ted = FuzzySet([0.6, 0.5, 0.3, 0.7, 0.3], [0.1, 0.4, 0.4, 0.2, 0.4])
    patients = [al, bob, joe, ted]

    return diagnoses, patients
    