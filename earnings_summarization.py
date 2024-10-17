# earnings_summarization.py
import streamlit as st
import pdfplumber
from fpdf import FPDF
import os
import tempfile
from transformers import pipeline

# Function to safely read PDF content using pdfplumber
@st.cache_data
def read_pdf(file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_file_path = tmp_file.name

        content = ""
        with pdfplumber.open(tmp_file_path) as pdf:
            for page in pdf.pages:
                content += page.extract_text() or ""

        os.unlink(tmp_file_path)  # Clean up the temporary file
        return content
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def display_earnings_page():
    st.title("Earnings Report Summarizer")

    # Step 1: Upload PDF (10-Q or 10-K)
    st.write("Upload 10-Q or 10-K PDF documents for summarization.")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    # Initialize session state for PDF content
    if 'pdf_content' not in st.session_state:
        st.session_state.pdf_content = None
    if uploaded_file is not None and uploaded_file != st.session_state.get('last_uploaded_file'):
        st.session_state.pdf_content = None
        st.session_state['last_uploaded_file'] = uploaded_file

    # Process the uploaded file
    if uploaded_file is not None and st.session_state.pdf_content is None:
        with st.spinner("Processing PDF..."):
            st.session_state.pdf_content = read_pdf(uploaded_file)

    # Ask user for their OpenAI API key
    st.markdown("### OpenAI API Key")
    openai_api_key = st.text_input("OpenAI API Key", type="password", key="api_key_input")

    if st.session_state.pdf_content is None:
        st.info("Please upload a 10-Q or 10-K PDF to proceed.")
    elif not openai_api_key:
        st.info("Please enter your OpenAI API key to proceed.")
    else:
        # Display the first 500 characters of the document for reference
        st.write("Extracted text from the document (first 500 characters):")
        st.text_area("", st.session_state.pdf_content[:500], height=200, disabled=True)
        
        # Step 3: Use Hugging Face transformers for summarization
        if st.button("Generate Report"):
            with st.spinner("Generating report... This might take a few moments."):
                try:
                    # Load a summarization model (e.g., BART)
                    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
                    input_text = st.session_state.pdf_content[:3000]  # Use the first 3000 characters
                    summary = summarizer(input_text, max_length=1024, min_length=300, do_sample=False)[0]['summary_text']
                    
                    st.success("Report generated successfully!")
                    st.markdown("### Summary")
                    st.write(summary)

                    # Step 4: Create a downloadable PDF report
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    pdf.multi_cell(0, 10, "Prospectus Report")
                    pdf.multi_cell(0, 10, summary)

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                        pdf.output(tmp_pdf.name)
                        tmp_pdf_path = tmp_pdf.name

                    with open(tmp_pdf_path, "rb") as f:
                        st.download_button("Download Report as PDF", f, "prospectus_report.pdf")

                    os.unlink(tmp_pdf_path)

                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")
