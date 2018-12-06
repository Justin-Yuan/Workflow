""" Centralized paths catalog 
"""

import os 


class DatasetCatalog(object):
    """ 
    """
    DATA_DIR = "datasets"

    DATASETS = {
        "name": (
            "data1",
            "data2"
        )
    }

    @staticmethod
    def get(name):
        args = None  # aggregate path names from DATASETS 
        return dict(factory="Dataset name", args=args)

    
class ModelCatalog(object):
    """
    """
    @staticmethod 
    def get(name):
        model_path = None 
        return model_path 