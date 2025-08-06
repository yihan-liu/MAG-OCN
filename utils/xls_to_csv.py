# xls_to_csv.py

import argparse

import pandas as pd

def xls_to_multiple_csv_format1(xlsx_path):
    new_header = ["ATOM", "X", "Y", "Z", "MAGNETIC_MOMENT"]
    
    # Read the Excel file to get sheet names
    xls = pd.ExcelFile(xlsx_path)
    
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xlsx_path, sheet_name=sheet_name, skiprows=1, header=None)
        df.columns = new_header
        csv_filename = f"{sheet_name}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"Sheet '{sheet_name}' has been saved to '{csv_filename}'")

def xls_to_multiple_csv_format2(xlsx_path):
    new_header = ["ATOM", "X", "Y", "Z", "MAGNETIC_MOMENT"]

    # Read the Excel file to get sheet names
    xls = pd.ExcelFile(xlsx_path)

    for sheet_name in xls.sheet_names:
        # Read the Excel sheet, skipping the first row (header)
        df = pd.read_excel(xlsx_path, sheet_name=sheet_name, skiprows=1, header=None)
        
        # Assume the columns are: [Index, Atom_Type, X, Y, Z, Magnetic_Moment]
        # Combine first two columns to create ATOM names like "N1", "N2", etc.
        df['ATOM'] = df.iloc[:, 1].astype(str) + df.iloc[:, 0].astype(str)
        
        # Extract coordinates and magnetic moment
        df['X'] = df.iloc[:, 2]
        df['Y'] = df.iloc[:, 3] 
        df['Z'] = df.iloc[:, 4]
        
        # Multiply magnetic moment by 1000 to convert units
        df['MAGNETIC_MOMENT'] = df.iloc[:, 5] * 1000
        
        # Select only the columns we need with the standard header
        result_df = df[['ATOM', 'X', 'Y', 'Z', 'MAGNETIC_MOMENT']]
        
        # Save to CSV
        csv_filename = f"{sheet_name}.csv"
        result_df.to_csv(csv_filename, index=False)
        print(f"Sheet '{sheet_name}' has been saved to '{csv_filename}'")

# Example usage
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', default=None,
                        help='Path to the xlsx file to convert.')
    parser.add_argument('--format', choices=['format1', 'format2'], default='format1',
                        help='Format of the CSV files to generate.')
    args = parser.parse_args()
    if args.format == 'format1':
        xls_to_multiple_csv_format1(args.filepath)
    else:
        xls_to_multiple_csv_format2(args.filepath)

