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
if __name__ == "__main__":
    path_to_excel = "original_dataset/CN-origin_Cartesian.xlsx"
    xlsx_to_multiple_csv(path_to_excel)

