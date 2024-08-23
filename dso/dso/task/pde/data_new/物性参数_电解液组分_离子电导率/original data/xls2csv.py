import openpyxl
import os


def read_excel(file_path):
    wb = openpyxl.load_workbook(file_path)
    ws = wb.active
    data = []
    for row in ws.iter_rows():
        data.append([cell.value for cell in row])
    return data


def remove_empty_cell(data):
    return [[cell for cell in row if cell is not None] for row in data]


def remove_empty_row(data):
    return [row for row in data if any(cell is not None for cell in row)]


def write_csv(data, file):
    with open(file, 'w', encoding='utf-8') as f:
        for row in data:
            f.write(','.join([str(cell) for cell in row]) + '\n')

files = os.listdir('.')
for file in files:
    if file.endswith('.xlsx'):
        data = read_excel(os.path.join('.', file))
        write_csv(remove_empty_row(remove_empty_cell(data)), os.path.join('.', file.replace('.xlsx', '.csv')))