from typing import List
import pandas as pd
import numbers

def table2tex(df: pd.DataFrame, title_cols: List[str] = None, bold_row_idx:int = None) -> List[str]:
    """Convert a pd.Dataframe to latex code

    Args:
        df (pd.DataFrame): table to be converted
        title_cols (list, optional): Title columns. Defaults to None.
        bold_row_idx (int, optional): Some rows that needs to have bold font. Defaults to None.

    Returns:
        _type_: _description_
    """
    if (title_cols is None):
        title_cols = list(df.columns)
    title_str = " & ".join([f"\\textbf{{{c.replace('_','-')}}}" for c in title_cols])
    final_s_list = ["\\hline \\hline", title_str + " \\\\", "\\hline"]
    for idx, row in df.iterrows():
        s = []
        for item in row:
            if (isinstance(item, numbers.Number)):
                s_tmp = f"{item:.3f}"
            else:
                s_tmp = item
            s.append(s_tmp)
        if (
            (isinstance(bold_row_idx, numbers.Number) and idx == bold_row_idx) or
            (isinstance(bold_row_idx, list) and idx in bold_row_idx)
        ):
            s = [f"\\textbf{{{s_}}}" for s_ in s]
        final_s_list.append(" & ".join(s) + " \\\\")
    final_s_list.append("\\hline \\hline")
    return [s for s in final_s_list]
    