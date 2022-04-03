from .sets import IntuitionisticFuzzySet


def load_patients_diagnoses():
    """ Returns diagnoses and patients data for the medical diagnosis application.

    The data returned are consisted of IntuitionisticFuzzySets representing the symptoms each
    patient has and the symptoms of each diagnosis.
    The symptoms include the following:
    **Temperature**, **Headache**, **Stomach paint**, **Cough** and **Chest paint**.

        
    Returns
    -------
    diagnoses : list[IntuitionisticFuzzySet]
        List of IntuitionisticFuzzySets representing each giadnosis: 
        **Viral fever**, **Malaria**, **Typhoid**, **Stomach problem** and **Chest problem**. 
    patients : list[IntuitionisticFuzzySet]
        List of IntuitionisticFuzzySets representing each patient's symptoms:
        Al, Bob, Joe and Ted.
    
    Examples
    --------
    Diagnoses:

    >>> viral_fever = IntuitionisticFuzzySet([0.4, 0.3, 0.1, 0.4, 0.1], [0.0, 0.5, 0.7, 0.3, 0.7])
    >>> malaria = IntuitionisticFuzzySet([0.7, 0.2, 0.0, 0.7, 0.1], [0.0, 0.6, 0.9, 0.0, 0.8])
    >>> typhoid = IntuitionisticFuzzySet([0.3, 0.6, 0.2, 0.2, 0.1], [0.3, 0.1, 0.7, 0.6, 0.9])
    >>> stomach_problem = IntuitionisticFuzzySet([0.1, 0.2, 0.8, 0.2, 0.2], [0.7, 0.4, 0.0, 0.7, 0.7])
    >>> chest_problem = IntuitionisticFuzzySet([0.1, 0.0, 0.2, 0.2, 0.8], [0.8, 0.8, 0.8, 0.8, 0.1])

    Patients:
    
    >>> al = IntuitionisticFuzzySet([0.8, 0.6, 0.2, 0.6, 0.1], [0.1, 0.1, 0.8, 0.1, 0.6])
    >>> bob = IntuitionisticFuzzySet([0.0, 0.4, 0.6, 0.1, 0.1], [0.8, 0.4, 0.1, 0.7, 0.8])
    >>> joe = IntuitionisticFuzzySet([0.8, 0.8, 0.0, 0.2, 0.0], [0.1, 0.1, 0.6, 0.7, 0.5])
    >>> ted = IntuitionisticFuzzySet([0.6, 0.5, 0.3, 0.7, 0.3], [0.1, 0.4, 0.4, 0.2, 0.4])
    """
    # diagnosis = [temperature, headache, stomach pain, cough, chest paint]
    viral_fever = IntuitionisticFuzzySet([0.4, 0.3, 0.1, 0.4, 0.1],
                           [0.0, 0.5, 0.7, 0.3, 0.7])
    malaria = IntuitionisticFuzzySet([0.7, 0.2, 0.0, 0.7, 0.1], [0.0, 0.6, 0.9, 0.0, 0.8])
    typhoid = IntuitionisticFuzzySet([0.3, 0.6, 0.2, 0.2, 0.1], [0.3, 0.1, 0.7, 0.6, 0.9])
    stomach_problem = IntuitionisticFuzzySet([0.1, 0.2, 0.8, 0.2, 0.2], [
                               0.7, 0.4, 0.0, 0.7, 0.7])
    chest_problem = IntuitionisticFuzzySet([0.1, 0.0, 0.2, 0.2, 0.8], [
                             0.8, 0.8, 0.8, 0.8, 0.1])
    diagnoses = [viral_fever, malaria, typhoid, stomach_problem, chest_problem]

    # patient = membership values for each characteristic, non-membership values for each characteristic
    al = IntuitionisticFuzzySet([0.8, 0.6, 0.2, 0.6, 0.1], [0.1, 0.1, 0.8, 0.1, 0.6])
    bob = IntuitionisticFuzzySet([0.0, 0.4, 0.6, 0.1, 0.1], [0.8, 0.4, 0.1, 0.7, 0.8])
    joe = IntuitionisticFuzzySet([0.8, 0.8, 0.0, 0.2, 0.0], [0.1, 0.1, 0.6, 0.7, 0.5])
    ted = IntuitionisticFuzzySet([0.6, 0.5, 0.3, 0.7, 0.3], [0.1, 0.4, 0.4, 0.2, 0.4])
    patients = [al, bob, joe, ted]

    return diagnoses, patients
    