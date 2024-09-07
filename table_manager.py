import pandas as pd
from datetime import datetime, timedelta

def create_table(course_data):
    table_data = [{
        'Topic': course_data['Topic'],
        'Organizer': course_data['Organizer'],
        'Start Time': course_data['Start Time'],
        'End Time': course_data['End Time'],
        'Location': course_data['Location'],
        'Topic Description': course_data['Topic Description'],
        'Speaker': course_data['Speaker'],
        'Duration': calculate_duration(course_data['Start Time'], course_data['End Time']),
        'Credit Category': determine_credit_category(course_data['Organizer']),
        'Credit Points': 0,  # Will be calculated later
        'AI Preliminary Review': '',  # Will be filled by AI review
        'AI Preliminary Review Credit Points': 0,  # Will be calculated later
        'Manual Review Confirmation': 'Pending'
    }]
    return table_data

def calculate_duration(start_time, end_time):
    if not start_time or not end_time:
        return 0  # Return 0 minutes if either start_time or end_time is empty
    try:
        start = datetime.strptime(start_time, "%Y-%m-%d %H:%M")
        end = datetime.strptime(end_time, "%Y-%m-%d %H:%M")
        duration = end - start
        return duration.total_seconds() / 60  # Duration in minutes
    except ValueError:
        print(f"Error parsing date strings: start_time={start_time}, end_time={end_time}")
        return 0  # Return 0 minutes if there's an error parsing the date strings

def determine_credit_category(organizer):
    if "Chinese Taipei Diabetes Association" in organizer or "Diabetes Association" in organizer:
        return "Category A"
    else:
        return "Category B"

def calculate_credits(table_data):
    for row in table_data:
        duration = row['Duration']
        category = row['Credit Category']
        
        if category == "Category A":
            credits = duration / 50
        else:  # Category B
            credits = (duration / 50) * 0.5
        
        row['Credit Points'] = round(credits, 1)
        row['AI Preliminary Review Credit Points'] = row['Credit Points']  # Initial value, may be adjusted later
    
    return table_data
