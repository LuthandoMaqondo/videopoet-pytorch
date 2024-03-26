import torch
import functools
from datetime import datetime

def format_datetime(input_datetime=datetime.now()):
    formatted_date = input_datetime.strftime("%d/%m/%Y at %H:%M:%S")
    return formatted_date


# helpers
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d