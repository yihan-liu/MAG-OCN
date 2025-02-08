# xls_to_csv.py

import argparse

import pandas as pd

def xlsx_to_multiple_csv(xlsx_path):
    # The desired new header
    new_header = ["ATOM", "X", "Y", "Z", "MAGNETIC_MOMENT"]
    
    # Read the Excel file to get sheet names
    xls = pd.ExcelFile(xlsx_path)
    
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xlsx_path, sheet_name=sheet_name, skiprows=1, header=None)
        df.columns = new_header
        csv_filename = f"{sheet_name}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"Sheet '{sheet_name}' has been saved to '{csv_filename}'")

# Example usage
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', default=None,
                        help='Path to the xlsx file to convert.')
    args = parser.parse_args()
    xlsx_to_multiple_csv(args.filepath)

