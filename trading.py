# -*- coding: utf-8 -*-
import pandas as pd


def preprocess(history_filename):
    df = pd.read_csv(history_filename)

    return df
