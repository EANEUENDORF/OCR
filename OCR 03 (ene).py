import os
import pytesseract
import pandas as pd
from pdf2image import convert_from_path

# Path to Tesseract-OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update with your path

def pdf_to_text(pdf_path):
    """
    Convert pdf file to string using OCR.

    Parameters:
    - pdf_path: str, path to the pdf file.

    Returns:
    - text: str, extracted text from pdf.
    """
    try:
        # Convert PDF to images
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = 500000000
        images = convert_from_path(pdf_path, 500) # 500 dpi to improve OCR quality

        # OCR processing on each image
        text = ''
        for i, img in enumerate(images):
            text += pytesseract.image_to_string(img)

        return text

    except Exception as e:
        print(f"Error in processing {pdf_path}: {str(e)}")
        return None

def batch_pdf_to_text(pdf_folder_path):
    """
    Process all PDFs in a folder and store OCR text in a DataFrame.

    Parameters:
    - pdf_folder_path: str, path to the folder containing pdf files.

    Returns:
    - df: DataFrame, contains filenames and their corresponding extracted text.
    """
    # Validate folder path
    if not os.path.exists(pdf_folder_path):
        raise FileNotFoundError(f"{pdf_folder_path} not found.")

    # Get all pdf files in folder
    pdf_files = [f for f in os.listdir(pdf_folder_path) if f.endswith(".pdf")]

    # Processing all PDFs and store the results
    data = {
        'filename': [],
        'text': []
    }

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder_path, pdf_file)
        text = pdf_to_text(pdf_path)
        if text is not None:
            data['filename'].append(pdf_file)
            data['text'].append(text)

    df = pd.DataFrame(data)
    return df

# Example usage
pdf_folder_path = 'T:/Projects/_Transer/Pitchbook 2023/Pitchbook Project/Test' # Update with your path
df = batch_pdf_to_text(pdf_folder_path)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_colwidth', 50000) 
print(df)

# Specify your desired file path
output_file_path = 'T:/Projects/_Transer/Pitchbook 2023/Pitchbook Project/Test/ocr_results.csv'

# Write the DataFrame to a CSV
df.to_csv(output_file_path, index=False)

# Optionally, save the results to a CSV file
# df.to_csv('ocr_results.csv', index=False)

headers = ["Header1", "Header2", "Header3", ...]



import pytesseract

def extract_text_with_positions(img):
    """
    Extract text from the provided image along with its position data.
    
    Parameters:
    - img: PIL image object
    
    Returns:
    - List of dictionaries containing word, x, y, width, and height.
    """
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    
    extracted_data = []
    for i in range(len(data['text'])):
        word = data['text'][i]
        if word:  # Only consider non-empty strings
            extracted_data.append({
                'word': word,
                'x': data['left'][i],
                'y': data['top'][i],
                'width': data['width'][i],
                'height': data['height'][i]
            })
            
    return extracted_data

# Use the function
from pdf2image import convert_from_path

# Example usage
image_path = 'T:/Projects/_Transer/Pitchbook 2023/Pitchbook Project/Test/1531.pdf'
# Convert PDF to images
from PIL import Image
Image.MAX_IMAGE_PIXELS = 500000000
images = convert_from_path(image_path, 500) # 500 dpi to improve OCR quality
img = images[0]  # Get the first (and only) image

# Extract position data
data = extract_text_with_positions(img)

# Print the extracted data
for item in data:
    print(f"Word: {item['word']}, X: {item['x']}, Y: {item['y']}, Width: {item['width']}, Height: {item['height']}")

import csv
# Specify your desired file path
output_file_path = 'T:/Projects/_Transer/Pitchbook 2023/Pitchbook Project/Test/ocr_results.csv'

# Write the extracted data to a CSV
with open(output_file_path, 'w', newline='') as csvfile:
    fieldnames = ['word', 'x', 'y', 'width', 'height']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()  # Write the headers (column names)
    for item in data:
        writer.writerow(item)

###################

from itertools import product
# Path to Tesseract-OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update with your path

def pdf_to_text_with_structure(pdf_path, headers):
    """
    Convert pdf file to a DataFrame using OCR and positional data.
    Parameters:
    - pdf_path: str, path to the pdf file.
    - headers: list, list of column headers to structure the data.
    Returns:
    - df: DataFrame, structured data from pdf.
    """
    try:
        # Convert PDF to images
        images = convert_from_path(pdf_path, 500)  # 500 dpi to improve OCR quality
        
        # Data placeholder
        all_columns_data = {header: [] for header in headers}
        
        # Define a vertical range for header detection
        y_range = 10  # Modify as needed
        
        # Iterate through each image
        for img in images:
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            
            column_positions = {}
            for header in headers:
                words = header.split()
                if len(words) > 1:
                    # For headers that consist of multiple words
                    word_indices = [[i for i, text in enumerate(data['text']) if text == word and abs(data['top'][i] - 787) <= y_range] for word in words]
                    
                    # Find the combination of words with the smallest distance and smallest x value for the first word
                    min_distance = float('inf')
                    min_x = float('inf')
                    chosen_indices = []
                    for combination in product(*word_indices):
                        distance = data['left'][combination[-1]] - data['left'][combination[0]]
                        if 0 < distance < min_distance or (distance == min_distance and data['left'][combination[0]] < min_x):
                            min_distance = distance
                            min_x = data['left'][combination[0]]
                            chosen_indices = combination

                    if chosen_indices:
                        column_positions[header] = data['left'][chosen_indices[0]]
                else:
                    # For headers that consist of a single word
                    header_indices = [i for i, text in enumerate(data['text']) if text == header and abs(data['top'][i] - 787) <= y_range]
                    if header_indices:
                        column_positions[header] = data['left'][header_indices[0]]
            
            columns_data = {header: [] for header in headers}
            
            sorted_headers = sorted(column_positions.items(), key=lambda x: x[1])  # Sort headers by their x-coordinate
            for i, word in enumerate(data['text']):
                word_x = data['left'][i]
                for j, (header, x_pos) in enumerate(sorted_headers):
                    # Check if it's the last header
                    if j == len(sorted_headers) - 1:
                        if word_x >= x_pos:
                            columns_data[header].append(word)
                    else:
                        # If word's x is between current header's x and the next header's x
                        if x_pos <= word_x < sorted_headers[j+1][1]:
                            columns_data[header].append(word)
            
            # Append this image's data to the overall data
            for header in headers:
                all_columns_data[header].extend(columns_data[header])
        
        # Handle empty cells
        max_len = max(len(col) for col in all_columns_data.values())
        for header in headers:
            while len(all_columns_data[header]) < max_len:
                all_columns_data[header].append(None)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_columns_data)
        return df
    except Exception as e:
        print(f"Error in processing {pdf_path}: {str(e)}")
        return
    

def batch_pdf_to_text(pdf_folder_path):
    """
    Process all PDFs in a folder and store OCR text in a DataFrame.

    Parameters:
    - pdf_folder_path: str, path to the folder containing pdf files.

    Returns:
    - df: DataFrame, contains filenames and their corresponding extracted text.
    """
    # Validate folder path
    if not os.path.exists(pdf_folder_path):
        raise FileNotFoundError(f"{pdf_folder_path} not found.")

    # Get all pdf files in folder
    pdf_files = [f for f in os.listdir(pdf_folder_path) if f.endswith(".pdf")]

    # Processing all PDFs and store the results
    data = {
        'filename': [],
        'text': []
    }

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder_path, pdf_file)
        text = pdf_to_text(pdf_path)
        if text is not None:
            data['filename'].append(pdf_file)
            data['text'].append(text)

    df = pd.DataFrame(data)
    return df

# Example usage
pdf_folder_path = 'T:/Projects/_Transer/Pitchbook 2023/Pitchbook Project/Test' # Update with your path
headers = ["Header1", "Header2", "Header3", ...]
df = batch_pdf_to_text(pdf_folder_path)