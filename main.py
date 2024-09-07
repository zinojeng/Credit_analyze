import streamlit as st
import pandas as pd
from document_parser import parse_document
from table_manager import create_table, calculate_credits
from ai_reviewer import perform_ai_review
from utils import export_table
import logging
from io import StringIO
import os

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add a StringIO handler to capture log output
log_stream = StringIO()
stream_handler = logging.StreamHandler(log_stream)
stream_handler.setLevel(logging.DEBUG)
logger.addHandler(stream_handler)

st.set_page_config(page_title="Medical Education Course Evaluator", layout="wide")

st.title("Medical Education Course Information Evaluator")

uploaded_files = st.file_uploader("Upload course documents (PDF, Word, JPEG)", 
                                  type=["pdf", "docx", "jpeg", "jpg"], 
                                  accept_multiple_files=True)

if uploaded_files:
    all_course_data = []

    for file in uploaded_files:
        st.subheader(f"Processing: {file.name}")
        logger.debug(f"Processing file: {file.name}")
        
        # Parse document and extract information
        course_data = parse_document(file)
        
        if course_data:
            logger.debug(f"Successfully parsed file: {file.name}")
            logger.debug(f"Extracted course data: {course_data}")
            # Create structured table
            table_data = create_table(course_data)
            
            # Calculate credits
            table_data = calculate_credits(table_data)
            
            all_course_data.extend(table_data)
        else:
            logger.error(f"Failed to parse file: {file.name}")
            st.error(f"Unable to extract information from {file.name}. Please check if the file is corrupted or in an unsupported format.")

    if all_course_data:
        logger.debug("Creating DataFrame from all course data")
        df = pd.DataFrame(all_course_data)
        
        # Perform AI preliminary review
        try:
            df = pd.DataFrame(perform_ai_review(df.to_dict('records')))
        except Exception as e:
            logger.error(f'Error during AI review: {str(e)}')
            st.error('An error occurred during the AI review process. Please check the logs for more information.')
        
        st.subheader("Course Information Table")
        st.dataframe(df)

        # Display AI Model Performance Report
        st.subheader('AI Model Performance Report')
        log_output = log_stream.getvalue()
        model_report = [line for line in log_output.split('\n') if 'AI Model Performance Report:' in line or any(metric in line for metric in ['Accuracy:', 'Precision:', 'Recall:', 'F1 Score:', 'Cross-validation Score:', 'Observations:', 'Potential Improvements:'])]
        if model_report:
            for line in model_report:
                if 'Observations:' in line or 'Potential Improvements:' in line:
                    st.subheader(line.strip())
                else:
                    st.write(line.strip())
        else:
            st.info('No AI Model Performance Report available. Please process more documents to generate a report.')

        # Calculate and display AI review accuracy
        ai_reviews = df['AI Preliminary Review'].tolist()
        manual_reviews = df['Manual Review Confirmation'].tolist()
        confirmed_reviews = [m for m in manual_reviews if m != 'Pending']
        if confirmed_reviews:
            accuracy = sum([1 for ai, manual in zip(ai_reviews, manual_reviews) if ai == manual and manual != 'Pending']) / len(confirmed_reviews)
            st.write(f"AI Review Accuracy (based on manual confirmations): {accuracy:.2%}")
        else:
            st.write("AI Review Accuracy: No manual confirmations available yet.")

        # Add progress bar for manual review confirmations
        total_reviews = len(manual_reviews)
        confirmed_count = len(confirmed_reviews)
        st.progress(confirmed_count / total_reviews)
        st.write(f"Manual Review Progress: {confirmed_count}/{total_reviews}")

        # Manual review confirmation
        st.subheader("Manual Review Confirmation")
        for index, row in df.iterrows():
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"Topic: {row['Topic']}")
            with col2:
                st.write(f"AI Review: {row['AI Preliminary Review']}")
                st.write(f"Confidence: {row['AI Review Confidence']}")
            with col3:
                confirmation = st.selectbox(
                    f"Confirm review for {row['Topic']}",
                    ["Pending", "Confirmed", "Rejected"],
                    key=f"confirm_{index}"
                )
                df.at[index, 'Manual Review Confirmation'] = confirmation

        # Export option
        if st.button("Export Table"):
            logger.debug("Exporting table")
            csv = export_table(df)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="course_information.csv",
                mime="text/csv"
            )
    else:
        logger.warning("No valid course data extracted from the uploaded files")
        st.warning("No valid course data extracted from the uploaded files.")
else:
    logger.info("Waiting for file upload")
    st.info("Please upload course documents to begin the evaluation process.")

logger.debug("Streamlit app finished running")
