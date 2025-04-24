from typing import *

from scikit_learn_bench import CONST


def sort_dict(d, ml_criteria):
    if ml_criteria == 0:
        return dict(sorted(d.items()))
    else:
        index = ml_criteria - 1
        return dict(sorted(d.items(), key=lambda item: item[1][index] if item[1][index] != CONST.NANSTR else -1))

def smart_round(number:float):
    """
    Round a number to a given number of significant digits.
    Strips trailing zeros, avoids scientific notation when possible.
    Example:
    CONST.ROUNDING=3 and CONST.THRESHOLD_DECIMAL = 100
    print(smart_round_sigfigs(12345.2546))  # 12345
    print(smart_round_sigfigs(0.012345))    # 0.012
    print(smart_round_sigfigs(1.2345))      # 1.23
    print(smart_round_sigfigs(123.456))     # 123
    print(smart_round_sigfigs(0.00056789.)) # 0.000568
    print(smart_round_sigfigs(999999.))     # 999999
    """
    if number == 0:
        return 0

    from math import log10, floor

    if number >= CONST.THRESHOLD_DECIMAL:
        return int(round(number))
    digits = CONST.ROUNDING - int(floor(log10(abs(number))))
    rounded = round(number, digits)
    return float(f"{rounded:.{CONST.ROUNDING}g}")

def print_table(score:Dict[str, Tuple], sorting_crit:int=-1)->None:
    if len(score) < 1:
        print("No score table to display")
        return

    # Sorting
    if sorting_crit!=-1:
        score = sort_dict(score, sorting_crit)

    # compute the number of chars per column
    num_cols = len(score[list(score.keys())[0]])+1
    number_of_chars_per_column = [0] * num_cols
    for algo, metrics in score.items():
        row=[algo]
        row.extend(list(metrics))
        for j, val in enumerate(row):
            nb_char_val = len(str(val))
            number_of_chars_per_column[j] = max(nb_char_val, number_of_chars_per_column[j])

    # Compute each row
    white_space_sep = 1
    rows = []
    for algo, metrics in score.items():
        row=[algo]
        row.extend(list(metrics))
        row_text = ""
        for j, val in enumerate(row):
            if isinstance(val,int) or isinstance(val,float):
                val_rounded=smart_round(val)
                val_str = str(val_rounded)
            else:
                val_str = val # E.g., "N/A"

            nb_white_space = ((number_of_chars_per_column[j] - len(val_str)) + white_space_sep)
            row_text += val_str
            row_text += " " * nb_white_space
        rows.append(row_text)

    # Print the rows
    for r in rows:
        print(r)
