import ratio
import auto
import pandas as pd
from flask import Flask, render_template_string

def merger():

    auto.main()

    ratio.main()

    # Load the data from the Excel and CSV files
    excel_file = 'auto_updated_with_decisions.xlsx'
    csv_file = 'put_call_ratio_data.csv'

    # Read the Excel file
    df_excel = pd.read_excel(excel_file)

    # Read the CSV file
    df_csv = pd.read_csv(csv_file)

    # Strip whitespace from column names (if any)
    df_excel.columns = df_excel.columns.str.strip()
    df_csv.columns = df_csv.columns.str.strip()

    # Drop rows with missing Stock Symbol but keep other entries
    df_excel = df_excel.dropna(subset=[df_excel.columns[0]], how='all')
    df_csv = df_csv.dropna(subset=[df_csv.columns[0]], how='all')

    # Prepare data for display using iloc
    # Selecting the 1st column (Stock Symbol) and the 8th column (index 7) from the Excel file
    df_excel_display = df_excel.iloc[:, [0, 6]].copy()  # 7 is the index for the 8th column
    df_excel_display.columns = ['Stock Symbol', 'Decision']  # Renaming for uniformity

    # Selecting the 1st column (Stock Symbol) and the 7th column (index 6) from the CSV file
    df_csv_display = df_csv.iloc[:, [0, 9]].copy()  # 6 is the index for the 7th column
    df_csv_display.columns = ['Stock Symbol', 'Final Decision']  # Renaming for uniformity

    # Merge data on 'Stock Symbol', keeping all entries from both sides
    merged_df = pd.merge(df_excel_display, df_csv_display, on='Stock Symbol', how='outer')
    # Apply the function to each row in the merged DataFrame to create a new column 'Final Output'
    merged_df['Final Output'] = merged_df.apply(determine_final_decision, axis=1)
    merged_df.to_csv('merged_output.csv', index=False)
    # Convert to a list of dictionaries for easier HTML rendering
    data_to_display = merged_df.to_dict(orient='records')
    return data_to_display

# Define a function to determine the final decision based on conditions
def determine_final_decision(row):
    
    excel_decision = row['Decision'] #tech
    csv_decision = row['Final Decision'] #pcr
    
    # If both decisions are SuperBuy
    if (excel_decision == 'SuperBuy' or excel_decision == 'IntraBuy') and csv_decision == 'SuperBuy':
        return 'SuperBuy'
    elif ((excel_decision == 'SuperBuy' or excel_decision == 'IntraBuy') and csv_decision == 'Sell') or (csv_decision == 'SuperBuy' and excel_decision == 'Sell'):
        return 'Watch'
    elif (excel_decision == 'SuperBuy' or csv_decision == 'SuperBuy') :
        return 'Buy'
    # If one is Buy and the other is Sell or Watch
    elif (excel_decision == 'SuperBuy' and csv_decision in ['Sell', 'Watch']) or (csv_decision == 'Buy' and excel_decision in ['Sell', 'Watch']):
        return 'Watch'
    # Otherwise, prefer the non-null value or leave as empty
    else:
        return excel_decision if pd.notnull(excel_decision) else csv_decision



# Create a simple Flask app to display the data
app = Flask(__name__)

# HTML template for rendering the data
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stock Decisions</title>
</head>
<body>
    <h1>Stock Decisions</h1>
    <table border="1">
        <tr>
            <th>Stock Symbol</th>
            <th>Technical</th>
            <th>PCR</th>
            <th>Final Output</th>
        </tr>
        {% for row in data %}
        <tr>
            <td>{{ row['Stock Symbol'] }}</td>
            <td>{{ row['Decision'] if row['Decision'] else '' }}</td>
            <td>{{ row['Final Decision'] if row['Final Decision'] else '' }}</td>
            <td>{{ row['Final Output'] if row['Final Output'] else '' }}</td>
        </tr>
        {% endfor %}
    </table>
</body>
</html>
"""

@app.route('/')
def index():
    #data_to_display = merger()
    return render_template_string(HTML_TEMPLATE, data=data_to_display)

if __name__ == '__main__':
    #app.run(debug=True)
    merger()