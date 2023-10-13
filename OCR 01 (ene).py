import os
import pytesseract
import pandas as pd
from pdf2image import convert_from_path

# Path to Tesseract-OCR
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update with your path

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
pdf_folder_path = 'path_to_your_pdf_folder' # Update with your path
df = batch_pdf_to_text(pdf_folder_path)

# Optionally, save the results to a CSV file
# df.to_csv('ocr_results.csv', index=False)