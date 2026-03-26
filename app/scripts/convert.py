import json
import nbformat
from pathlib import Path

# Obtener la ruta del directorio donde se encuentra este script (convert.py)
BASE_PATH = Path(__file__).resolve().parent

# Definir las rutas de entrada y salida usando la ruta base
input_file = BASE_PATH / 'kaggle_ocr_notebook_ipynb.json'
output_file = BASE_PATH / 'output.ipynb'

# Leer el archivo JSON
try:
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Crear un nuevo notebook
    nb = nbformat.v4.new_notebook()

    # Añadir celdas desde el JSON
    nb.cells = []
    for cell_data in data.get('cells', []):
        # Tomar el 'source' del JSON. Si es una lista, unirlo en un string.
        source = cell_data.get('source', "")
        if isinstance(source, list):
            source = "".join(source)
            
        cell_type = cell_data.get('cell_type', 'code')
        
        if cell_type == 'code':
            cell = nbformat.v4.new_code_cell(source)
        else:
            cell = nbformat.v4.new_markdown_cell(source)
            
        nb.cells.append(cell)

    # Guardar como archivo IPYNB
    with open(output_file, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
        
    print(f"✅ Notebook creado exitosamente en: {output_file}")

except FileNotFoundError:
    print(f"❌ Error: No se encontró el archivo en {input_file}")
except Exception as e:
    print(f"❌ Ocurrió un error: {e}")