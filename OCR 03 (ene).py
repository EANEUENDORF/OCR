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
from PIL import Image

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
        Image.MAX_IMAGE_PIXELS = 500000000
        images = convert_from_path(pdf_path, 500)  # 500 dpi to improve OCR quality
        
        # Data placeholder
        all_columns_data = {header: [] for header in headers}
        
        # Define a vertical range for header detection
        y_range = 2  # Modify as needed
        
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
            
            # Exclude words above the header line
            header_y_position = 787
            valid_indices = [i for i, y in enumerate(data['top']) if y > header_y_position-y_range]

            # Step 1: Using the sorted headers by their x-coordinates.
            sorted_headers = sorted(column_positions.items(), key=lambda x: x[1])

            # Step 2: Group words by their y-coordinate to form rows.
            grouped_by_y = {}
            for i in valid_indices:
                y = data['top'][i]
                found_group = False
                for existing_y in grouped_by_y.keys():
                    if abs(existing_y - y) <= y_range:
                        grouped_by_y[existing_y].append(i)
                        found_group = True
                        break

                if not found_group:
                    grouped_by_y[y] = [i]
                    
            rows_data = []
            for y, indices in grouped_by_y.items():
                row_data = {}
                for index in indices:
                    word = data['text'][index]
                    word_x = data['left'][index]
                    for j, (header, x_pos) in enumerate(sorted_headers):
                        if j == len(sorted_headers) - 1:
                            if word_x >= x_pos:
                                row_data[header] = word
                        else:
                            if x_pos <= word_x < sorted_headers[j+1][1]:
                                row_data[header] = word
                rows_data.append(row_data)

            # Append this image's data to the overall data
            for row in rows_data:
                for header in headers:
                    all_columns_data[header].append(row.get(header, None))

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
    

# Example usage
pdf_folder_path = 'T:/Projects/_Transer/Pitchbook 2023/Pitchbook Project/Test' # Update with your path
pdf_path = 'T:/Projects/_Transer/Pitchbook 2023/Pitchbook Project/Test/1531.pdf'
headers = ["#","Companies", "Ownership", "Description","Deal Type","Deal Type 2","Deal Date","Pre-money Valuation","Raised to Date","Deal Size","Revenue","Employees","Investors","Lead/Sole Investors","HQ Location","Financing Status","Business Status","Primary Industry","Verticals","Deal Status","Company Website"]

df = pdf_to_text_with_structure(pdf_path,headers)

# Specify your desired file path
output_file_path = 'T:/Projects/_Transer/Pitchbook 2023/Pitchbook Project/Test/ocr_results.csv'

# Write the DataFrame to a CSV
df.to_csv(output_file_path, index=False)

# Set the display options
pd.set_option('display.max_columns', None)  # Display all columns
pd.set_option('display.max_rows', None)     # Display all rows
pd.set_option('display.width', None)        # Ensure that the entire width of columns is displayed
pd.set_option('display.max_colwidth', -1)   # Display complete content in each cell

print(df)