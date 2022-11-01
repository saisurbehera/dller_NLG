'''
Saisamrit Surbehera
Nov 1 2022

This file contains the utility functions for the project. 
'''

from typing import string
from datasets import load_from_disk , load_dataset

def get_streaming_qa(file_loc : string) -> dataset:
    '''
    This function returns the streaming qa from the file location
    '''
    return open(file_loc, 'r').read()