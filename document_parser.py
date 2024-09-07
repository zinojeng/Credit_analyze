import pdfplumber
import docx
from PIL import Image
import pytesseract
import io
import streamlit as st
import logging
import re
import jieba
from datetime import datetime

logger = logging.getLogger(__name__)

def parse_document(file):
    file_extension = file.name.split('.')[-1].lower()
    
    if file_extension == 'pdf':
        course_info = parse_pdf(file)
    elif file_extension == 'docx':
        course_info = parse_docx(file)
    elif file_extension in ['jpeg', 'jpg']:
        course_info = parse_image(file)
    else:
        course_info = None
    
    if course_info:
        logger.debug(f"Extracted course info: {course_info}")
    else:
        logger.error(f"Failed to parse file: {file.name}")
    
    return course_info

def parse_pdf(file):
    try:
        with pdfplumber.open(file) as pdf:
            text = ""
            for page in pdf.pages:
                text += handle_multi_column(page)
        
        if not text.strip():
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    img = page.to_image()
                    text += pytesseract.image_to_string(img.original, lang='chi_sim+eng')
        
        return extract_course_info(text)
    except Exception as e:
        logger.error(f"Error parsing PDF: {str(e)}")
        return None

def handle_multi_column(page):
    text = page.extract_text()
    if text:
        return text
    
    height = page.height
    width = page.width
    left_column = page.crop((0, 0, width/2, height)).extract_text()
    right_column = page.crop((width/2, 0, width, height)).extract_text()
    return left_column + "\n" + right_column

def parse_docx(file):
    try:
        doc = docx.Document(file)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return extract_course_info(text)
    except Exception as e:
        logger.error(f"Error parsing DOCX: {str(e)}")
        return None

def parse_image(file):
    try:
        image = Image.open(file)
        text = pytesseract.image_to_string(image, lang='chi_sim+eng')
        return extract_course_info(text)
    except Exception as e:
        logger.error(f"Error parsing image: {str(e)}")
        return None

def extract_course_info(text):
    logger.debug("Starting course info extraction")
    info = {
        'Topic': extract_field(text, r'(Topic|主題|課程名稱)[:：]\s*(.+)'),
        'Organizer': extract_field(text, r'(Organizer|主辦單位)[:：]\s*(.+)'),
        'Start Time': extract_time(text, start=True),
        'End Time': extract_time(text, end=True),
        'Location': extract_field(text, r'(Location|地點)[:：]\s*(.+)'),
        'Topic Description': extract_field(text, r'(Topic Description|課程描述)[:：]\s*(.+)'),
        'Speaker': extract_speaker(text),
        'Moderator': extract_moderator(text),
        'Time': extract_time_range(text)
    }
    
    # Use jieba for more accurate text segmentation
    segmented_text = ' '.join(jieba.cut(text))
    
    # If regular expressions fail, try to find the closest match
    for key in info:
        if not info[key]:
            info[key] = find_closest_match(segmented_text, key)
            logger.debug(f"Closest match for {key}: {info[key]}")
    
    # Post-processing
    for key in ['Start Time', 'End Time']:
        info[key] = normalize_date_time(info[key])
    
    # Log extracted information for debugging
    for key, value in info.items():
        logger.debug(f"Extracted {key}: {value}")
    
    return info

def extract_field(text, pattern):
    match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
    return match.group(2).strip() if match else ''

def extract_time(text, start=True, end=False):
    time_type = "開始" if start else "結束"
    patterns = [
        rf'({time_type}|起始|開課|完成).*?時間[:：]\s*(.+)',
        r'(\d{4})年(\d{1,2})月(\d{1,2})日\s*(\d{1,2})[.:：](\d{2})',
        r'(\d{4})[-.／](\d{1,2})[-.／](\d{1,2})\s*(\d{1,2})[.:：](\d{2})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            if len(match.groups()) == 2:  # First pattern
                return match.group(2).strip()
            elif len(match.groups()) == 5:  # Date and time patterns
                year, month, day, hour, minute = match.groups()
                return f"{year}-{month.zfill(2)}-{day.zfill(2)} {hour.zfill(2)}:{minute}"
    
    logger.warning(f"Failed to extract {time_type} time using regular expressions")
    return ''

def extract_speaker(text):
    patterns = [
        r'(Speaker|講者)[:：]\s*(.+)',
        r'([^\s]+\s+[^\s]+)\s+(醫師|教授|博士)'
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(2).strip() if len(match.groups()) > 1 else match.group(1).strip()
    return ''

def extract_moderator(text):
    patterns = [
        r'(Moderator|主持人|主持)[:：]\s*(.+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(2).strip()
    return ''

def extract_time_range(text):
    pattern = r'\d{2}:\d{2}-\d{2}:\d{2}'
    match = re.search(pattern, text)
    return match.group(0) if match else ''

def find_closest_match(text, field):
    field_translations = {
        'Topic': '主題',
        'Organizer': '主辦單位',
        'Start Time': '開始時間',
        'End Time': '結束時間',
        'Location': '地點',
        'Topic Description': '課程描述',
        'Speaker': '講者',
        'Moderator': '主持人',
        'Time': '時間'
    }
    
    field_cn = field_translations.get(field, field)
    words = text.split()
    try:
        field_index = words.index(field_cn)
        return ' '.join(words[field_index+1:field_index+6])  # Return next 5 words
    except ValueError:
        logger.warning(f"Failed to find closest match for field: {field}")
        return ''

def normalize_date_time(date_time_str):
    try:
        # Try parsing with various formats
        for fmt in ('%Y-%m-%d %H:%M', '%Y年%m月%d日 %H:%M', '%Y/%m/%d %H:%M', '%m/%d/%Y %H:%M', '%d/%m/%Y %H:%M'):
            try:
                dt = datetime.strptime(date_time_str, fmt)
                return dt.strftime('%Y-%m-%d %H:%M')
            except ValueError:
                continue
        
        logger.warning(f"Failed to normalize date and time: {date_time_str}")
        return date_time_str
    except Exception as e:
        logger.error(f"Error normalizing date and time: {str(e)}")
        return date_time_str

# Initialize jieba
jieba.initialize()
