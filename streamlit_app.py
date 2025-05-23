import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import difflib
from typing import List, Tuple, Dict, Optional
import unicodedata
import base64 # Not directly used, but kept from original
from pathlib import Path # Not directly used, but kept from original

# Document processing imports
import docx
import PyPDF2
from markdown import markdown
from bs4 import BeautifulSoup

# ML imports for Chinese text processing
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
import plotly.graph_objects as go
from plotly.subplots import make_subplots # Not directly used, but kept
import plotly.express as px
import logging # For setting jieba log level

# Configure page
st.set_page_config(
    page_title="Chinese Text Alignment Tool",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

class ChineseTextProcessor:
    
    @staticmethod
    def normalize_text(text: str) -> str:
        text = unicodedata.normalize('NFC', text)
        text = re.sub(r'[\u3000\uFEFF\u200B-\u200D\u2060]', '', text) # Remove various zero-width and full-width spaces
        text = re.sub(r'[ \t]+', ' ', text) # Normalize horizontal whitespace
        text = re.sub(r'\n[ \t]*\n', '\n\n', text) # Normalize multiple newlines to double (preserves single newlines for segmentation)
        return text.strip()
    
    @staticmethod
    def remove_punctuation(text: str) -> str:
        # This pattern includes Chinese and common English punctuation, as well as whitespace including newlines
        punctuation_pattern = r'[\s,.!?;:()\[\]{}\'"、，。！？；：（）【】《》""''\\-—_\n]'
        text = re.sub(punctuation_pattern, '', text)
        return text
    
    @staticmethod
    def extract_text_from_file(uploaded_file) -> str:
        file_type = uploaded_file.name.split('.')[-1].lower()
        
        if file_type == 'txt':
            # Try to decode as UTF-8, with fallback for robustness
            try:
                return str(uploaded_file.read(), 'utf-8')
            except UnicodeDecodeError:
                uploaded_file.seek(0) # Reset file pointer
                return str(uploaded_file.read(), 'gbk', errors='replace') # Common encoding for Chinese texts
        elif file_type == 'docx':
            doc = docx.Document(uploaded_file)
            return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        elif file_type == 'pdf':
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ''
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text: # Ensure text was extracted
                    text += page_text + '\n'
            return text
        elif file_type == 'md':
            try:
                markdown_text = str(uploaded_file.read(), 'utf-8')
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                markdown_text = str(uploaded_file.read(), 'gbk', errors='replace')
            html = markdown(markdown_text)
            return BeautifulSoup(html, 'html.parser').get_text()
        else:
            st.error(f"Unsupported file type: {file_type}. Please upload TXT, DOCX, PDF, or MD files.")
            raise ValueError(f"Unsupported file type: {file_type}")

class ChineseSegmenter:
    
    def segment_text(self, text: str, min_length: int = 10) -> List[str]:
        text = ChineseTextProcessor.normalize_text(text)
        # Split by major punctuation and newlines, keeping the delimiters
        # The pattern ensures that sequences of delimiters are captured together
        split_pattern = r'([。！？!?；\n]+)' 
        parts = re.split(split_pattern, text)
        
        segments = []
        current_segment = ""
        
        i = 0
        while i < len(parts):
            part = parts[i].strip() 
            if not part: # Skip empty parts that might result from re.split
                i += 1
                continue
            
            # Check if the part is purely a delimiter sequence
            if re.fullmatch(r'[。！？!?；\n]+', part): # Use fullmatch to ensure the entire part is a delimiter
                current_segment += part # Append delimiter to current segment
                # If current segment (text + delimiter) is long enough, add it
                # Or if adding delimiter makes it long enough
                if len(ChineseTextProcessor.remove_punctuation(current_segment).strip()) >= min_length:
                    segments.append(current_segment.strip())
                    current_segment = ""
            else: # Part is text
                # If current_segment has content and adding this part creates a new semantic unit
                # (e.g., previous was just a delimiter, now new text starts),
                # and if current_segment was already substantial, consider adding it.
                # However, the logic tends to accumulate until a delimiter makes it "complete" or long enough.

                current_segment += (" " + part if current_segment else part) # Add space if appending to existing text

                # Heuristic to split very long segments without punctuation
                # Check length of text content only
                if len(ChineseTextProcessor.remove_punctuation(current_segment)) > min_length * 5: # Increased multiplier
                    segments.append(current_segment.strip())
                    current_segment = ""
            
            i += 1
        
        # Add any remaining segment
        if current_segment.strip():
            # If the last segment is too short and there are previous segments, append it to the last one.
            if segments and len(ChineseTextProcessor.remove_punctuation(current_segment).strip()) < min_length:
                segments[-1] += (" " + current_segment.strip())
            else:
                segments.append(current_segment.strip())
        
        # Final filter and cleanup: ensure all segments meet min_length for their text content
        final_segments = []
        temp_buffer = ""
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
            
            temp_buffer += (" " + segment if temp_buffer else segment)
            # Check content length without punctuation
            if len(ChineseTextProcessor.remove_punctuation(temp_buffer).strip()) >= min_length:
                final_segments.append(temp_buffer.strip())
                temp_buffer = ""
        
        # If there's anything left in temp_buffer (likely a short trailing segment)
        if temp_buffer.strip():
            if final_segments and len(ChineseTextProcessor.remove_punctuation(temp_buffer).strip()) < min_length:
                final_segments[-1] += (" " + temp_buffer.strip()) # Append to the last valid segment
            elif len(ChineseTextProcessor.remove_punctuation(temp_buffer).strip()) >= min_length : # Or if it's standalone valid
                 final_segments.append(temp_buffer.strip())
            # If it's short and no previous segment, it might be dropped unless it's the only thing

        # If final_segments is empty and the original text had content,
        # but was shorter than min_length, add it as a single segment.
        if not final_segments and ChineseTextProcessor.remove_punctuation(text).strip():
             if len(ChineseTextProcessor.remove_punctuation(text).strip()) > 0 : # Ensure there's some content
                final_segments.append(text.strip())

        return [s.strip() for s in final_segments if s.strip()]
    
    def add_segment_ids(self, segments: List[str], doc_id: str) -> List[Dict]:
        return [
            {
                'id': f"{doc_id}{(i+1):03d}", # Use 3 digits for ID for more segments
                'text': segment,
                'doc': doc_id,
                'index': i
            }
            for i, segment in enumerate(segments)
        ]

class ChineseSimilarityMatcher:
    
    def __init__(self):
        # Suppress verbose output from jieba
        jieba.setLogLevel(logging.INFO) # Or logging.WARNING / logging.ERROR

        def chinese_analyzer(text):
            return jieba.lcut(text)
        
        self.vectorizer = TfidfVectorizer(analyzer=chinese_analyzer, min_df=1) # min_df=1 for small docs
        self.model = None # Placeholder, not used in this TF-IDF version
    
    def find_best_matches(self, 
                     segments_a: List[Dict], 
                     segments_b: List[Dict], 
                     threshold: float = 0.6) -> List[Dict]:
        
        if not segments_a or not segments_b:
            # Handle cases where one or both segment lists are empty
            alignment_table = []
            for i, seg_a in enumerate(segments_a):
                 alignment_table.append({
                    'segment_a': seg_a, 'segment_b': None, 'similarity': 0.0,
                    'a_index': i, 'b_index': -1, 'matched': False
                })
            for j, seg_b in enumerate(segments_b):
                alignment_table.append({
                    'segment_a': None, 'segment_b': seg_b, 'similarity': 0.0,
                    'a_index': -1, 'b_index': j, 'matched': False
                })
            return alignment_table

        texts_a = [seg['text'] for seg in segments_a]
        texts_b = [seg['text'] for seg in segments_b]
        
        # Use punctuation-removed text for TF-IDF calculation for more robust similarity
        clean_texts_a = [ChineseTextProcessor.remove_punctuation(text) for text in texts_a]
        clean_texts_b = [ChineseTextProcessor.remove_punctuation(text) for text in texts_b]
        
        # Filter out empty strings that might result from remove_punctuation if a segment was only punctuation
        # Keep track of original indices to map back
        
        # Handle empty lists after cleaning
        if not any(clean_texts_a) and not any(clean_texts_b): # if all segments become empty after punc removal
            alignment_table = []
            processed_indices_a = set()
            for i, seg_a in enumerate(segments_a):
                alignment_table.append({
                    'segment_a': seg_a, 'segment_b': None, 'similarity': 0.0,
                    'a_index': i, 'b_index': -1, 'matched': False
                })
                processed_indices_a.add(i)
            
            for j, seg_b in enumerate(segments_b):
                alignment_table.append({
                    'segment_a': None, 'segment_b': seg_b, 'similarity': 0.0,
                    'a_index': -1, 'b_index': j, 'matched': False
                })
            return alignment_table


        all_clean_texts = [text for text in clean_texts_a if text] + [text for text in clean_texts_b if text]
        
        if not all_clean_texts: # If all texts are empty after cleaning
            # Fallback: create a table with no matches
            alignment_table = []
            for i, seg_a in enumerate(segments_a):
                 alignment_table.append({
                    'segment_a': seg_a, 'segment_b': None, 'similarity': 0.0,
                    'a_index': i, 'b_index': -1, 'matched': False})
            for j, seg_b in enumerate(segments_b):
                 alignment_table.append({
                    'segment_a': None, 'segment_b': seg_b, 'similarity': 0.0,
                    'a_index': -1, 'b_index': j, 'matched': False})
            return alignment_table

        self.vectorizer.fit(all_clean_texts)
        
        vectors_a = self.vectorizer.transform(clean_texts_a).toarray() if any(clean_texts_a) else np.array([]).reshape(len(clean_texts_a),0)
        vectors_b = self.vectorizer.transform(clean_texts_b).toarray() if any(clean_texts_b) else np.array([]).reshape(len(clean_texts_b),0)

        # Ensure vectors are not empty before cosine_similarity
        if vectors_a.shape[1] == 0 or vectors_b.shape[1] == 0: # No common features or one set is all empty
            if vectors_a.shape[0] > 0 and vectors_b.shape[0] > 0 and vectors_a.shape[1] != vectors_b.shape[1] and (vectors_a.shape[1] == 0 or vectors_b.shape[1] == 0):
                 # If one is valid and the other became all empty strings
                 similarity_matrix = np.zeros((vectors_a.shape[0], vectors_b.shape[0]))
            elif vectors_a.shape[0] == 0 or vectors_b.shape[0] == 0: # one of the original segment lists was empty
                 similarity_matrix = np.array([]) # or handle as appropriate
            else: # Both became empty after cleaning, or vocab issue
                 similarity_matrix = np.zeros((len(segments_a), len(segments_b)))
        elif vectors_a.size == 0 or vectors_b.size == 0: # One of the lists was empty to begin with
            similarity_matrix = np.array([])
        else:
            try:
                similarity_matrix = cosine_similarity(vectors_a, vectors_b)
            except ValueError: # Handles cases like one vector being all zeros if min_df is too high or text is too sparse
                 similarity_matrix = np.zeros((vectors_a.shape[0], vectors_b.shape[0]))

        alignment_table = []
        used_b_indices = set()
        
        # Greedily match A to B
        for i, seg_a in enumerate(segments_a):
            best_score = -1.0
            best_j = -1
            
            if similarity_matrix.size > 0 and i < similarity_matrix.shape[0]: # Check if row i exists
                for j in range(len(segments_b)):
                    if j in used_b_indices:
                        continue
                    if j < similarity_matrix.shape[1]: # Check if col j exists
                        score = similarity_matrix[i, j]
                        if score >= threshold and score > best_score:
                            best_score = score
                            best_j = j
            
            if best_j != -1:
                alignment_table.append({
                    'segment_a': seg_a,
                    'segment_b': segments_b[best_j],
                    'similarity': best_score,
                    'a_index': i,
                    'b_index': best_j,
                    'matched': True
                })
                used_b_indices.add(best_j)
            else:
                alignment_table.append({
                    'segment_a': seg_a,
                    'segment_b': None, # No match found or below threshold
                    'similarity': 0.0, # Or best_score if you want to show the highest score below threshold
                    'a_index': i,
                    'b_index': -1,
                    'matched': False
                })
        
        # Add unmatched B segments
        for j, seg_b in enumerate(segments_b):
            if j not in used_b_indices:
                alignment_table.append({
                    'segment_a': None,
                    'segment_b': seg_b,
                    'similarity': 0.0,
                    'a_index': -1,
                    'b_index': j,
                    'matched': False
                })
        
        return alignment_table

class CharacterGridAligner:
    
    @staticmethod
    def align_texts(text1: str, text2: str) -> Tuple[List[str], List[str]]:
        # Use punctuation-removed texts for generating diff opcodes
        clean1 = ChineseTextProcessor.remove_punctuation(text1)
        clean2 = ChineseTextProcessor.remove_punctuation(text2)
        
        if not clean1 and not clean2: # Both empty after punc removal
            return list(text1), list(text2) # Show original chars if clean versions are empty
        
        matcher = difflib.SequenceMatcher(None, clean1, clean2, autojunk=False)
        opcodes = matcher.get_opcodes()
        
        aligned1_chars = []
        aligned2_chars = []
        
        # These point to original texts
        ptr1, ptr2 = 0, 0
        # These point to clean texts
        clean_ptr1, clean_ptr2 = 0, 0

        # Helper to get next char from original text, skipping punctuation
        def get_next_original_char_and_advance(original_text, current_ptr, target_clean_char_index, clean_text_segment):
            char_to_return = ''
            # Consume original text until we've emitted the character corresponding to clean_text_segment[target_clean_char_index]
            # or until we hit punctuation that should be displayed.
            temp_collected_chars = []
            
            # This simplified logic may not perfectly re-insert all punctuation in complex cases.
            # The core idea is to advance in original text while matching clean text.
            
            # For this simplified version, let's just use the clean characters directly
            # and then try to reconstruct with original characters in the HTML part.
            # The align_texts will return aligned *clean* characters.
            # The create_grid_html will then map these back to original with punctuation.
            # This was the original design and is simpler to manage here.
            pass # Keeping this structure for now as per original code block

        for tag, i1, i2, j1, j2 in opcodes:
            if tag == 'equal':
                aligned1_chars.extend(list(clean1[i1:i2]))
                aligned2_chars.extend(list(clean2[j1:j2]))
            elif tag == 'delete': # In clean1 but not clean2
                aligned1_chars.extend(list(clean1[i1:i2]))
                aligned2_chars.extend([''] * (i2 - i1))
            elif tag == 'insert': # In clean2 but not clean1
                aligned1_chars.extend([''] * (j2 - j1))
                aligned2_chars.extend(list(clean2[j1:j2]))
            elif tag == 'replace':
                len_a = i2 - i1
                len_b = j2 - j1
                max_len = max(len_a, len_b)
                
                for k in range(max_len):
                    if k < len_a:
                        aligned1_chars.append(clean1[i1 + k])
                    else:
                        aligned1_chars.append('')
                    
                    if k < len_b:
                        aligned2_chars.append(clean2[j1 + k])
                    else:
                        aligned2_chars.append('')
        
        return aligned1_chars, aligned2_chars
    
    @staticmethod
    def get_character_status(char1_clean: str, char2_clean: str) -> str:
        if char1_clean == char2_clean and char1_clean != '':
            return 'equal'
        elif char1_clean == '' and char2_clean != '': # char1_clean is from aligned_a, char2_clean from aligned_b
            return 'insert' # char is in b, not in a (insertion into a to become b) -> Doc B unique
        elif char2_clean == '' and char1_clean != '':
            return 'delete' # char is in a, not in b (deletion from a to become b) -> Doc A unique
        else: # Both are non-empty and different
            return 'replace'
    
    @staticmethod
    def create_grid_html(alignment_data: List[Dict], chars_per_line: int = 20) -> str:
        # CSS for the grid
        html = """
        <!DOCTYPE html>
        <html lang="zh">
        <head>
            <meta charset="UTF-8">
            <title>中文字符对齐网格</title>
            <style>
                body {
                    font-family: "Source Han Serif SC", "Noto Serif CJK SC", "SimSun", serif;
                    background: #f0f2f5; /* Light gray background for page */
                    margin: 0;
                    padding: 20px;
                    line-height: 1.2; /* Compact line height */
                }
                .container {
                    max-width: 95%; /* Use more width */
                    margin: 0 auto;
                    background: white;
                    border-radius: 8px;
                    padding: 20px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                }
                h1 {
                    text-align: center;
                    color: #333;
                    margin-bottom: 20px;
                    font-size: 1.6em;
                }
                .segment-pair {
                    margin-bottom: 25px;
                    border: 1px solid #d9d9d9;
                    border-radius: 6px;
                    overflow: hidden;
                }
                .segment-header {
                    background: #4a5568; /* Darker header */
                    color: white;
                    padding: 10px 15px;
                    font-weight: bold;
                    font-size: 0.9em;
                }
                .grid-container {
                    padding: 10px;
                    background: #fafafa; /* Light background for grid area */
                }
                .row-pair { /* Contains a row for Doc A and a row for Doc B */
                    margin-bottom: 5px; /* Compact spacing between pairs of rows */
                }
                .grid-row {
                    display: flex;
                    margin-bottom: 1px; /* Very compact spacing between Doc A and Doc B rows */
                    align-items: stretch; /* Make cells in a row same height */
                }
                .row-label {
                    min-width: 50px; /* Min-width for label */
                    width: auto; /* Allow label to take space it needs */
                    padding: 0 5px;
                    font-weight: bold;
                    color: #4a5568;
                    font-size: 0.75em;
                    text-align: center;
                    margin-right: 5px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    background: #e2e8f0; /* Label background */
                    border-radius: 3px;
                }
                .char-cell {
                    width: 28px; /* Square cells */
                    height: 28px; /* Square cells */
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    border: 1px solid #d1d5db; /* Lighter border */
                    font-size: 14px; /* Adjust as needed */
                    font-weight: normal;
                    margin-right: 1px; /* Small gap between cells */
                    box-sizing: border-box;
                    font-family: "SimSun", "MS Mincho", "MingLiU", serif; /* Fonts good for CJK characters */
                    overflow: hidden; /* Prevent character overflow */
                    text-overflow: ellipsis;
                    white-space: nowrap;
                }
                .equal { background: #e6ffed; color: #2f5233; border-color: #a7e0b4; }
                .insert { background: #ffebe6; color: #7d2a17; border-color: #f5b8a9; } /* Doc B unique */
                .delete { background: #fff9e6; color: #806f19; border-color: #f7e8aa; } /* Doc A unique */
                .replace { background: #efedff; color: #493fb3; border-color: #c8c2f2; }
                .empty { background: #f8f9fa; border-color: #dee2e6; color: #adb5bd; }
                
                .legend { display: flex; justify-content: center; gap: 12px; margin: 15px 0; flex-wrap: wrap; }
                .legend-item { display: flex; align-items: center; gap: 5px; font-size: 0.8em; }
                .legend-color { width: 15px; height: 15px; border-radius: 3px; border: 1px solid #ccc; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>中文字符对齐网格视图</h1>
                <div class="legend">
                    <div class="legend-item"><div class="legend-color equal"></div><span>相同字符</span></div>
                    <div class="legend-item"><div class="legend-color insert"></div><span>文档B独有</span></div>
                    <div class="legend-item"><div class="legend-color delete"></div><span>文档A独有</span></div>
                    <div class="legend-item"><div class="legend-color replace"></div><span>字符不同</span></div>
                    <div class="legend-item"><div class="legend-color empty"></div><span>空位</span></div>
                </div>
        """
        
        # This mapping is tricky. The original code used clean text for alignment, then displayed original.
        # For simplicity, let's stick to aligning and displaying based on clean characters from `align_texts`.
        # Displaying original text with perfect punctuation re-insertion alongside diff markers is complex.
        # The current approach ensures aligned characters are shown clearly.

        for data in alignment_data:
            if not data.get('matched', False) or not data['segment_a'] or not data['segment_b']:
                continue
                
            seg_a_orig_text = data['segment_a']['text']
            seg_b_orig_text = data['segment_b']['text']
            similarity = data['similarity']
            
            # aligned_clean_a/b are lists of single characters or empty strings
            aligned_clean_a, aligned_clean_b = CharacterGridAligner.align_texts(seg_a_orig_text, seg_b_orig_text)
            
            html += f"""
            <div class="segment-pair">
                <div class="segment-header">
                    段落对比: {data['segment_a']['id']} ({len(ChineseTextProcessor.remove_punctuation(seg_a_orig_text))}字) ↔ {data['segment_b']['id']} ({len(ChineseTextProcessor.remove_punctuation(seg_b_orig_text))}字) | 相似度: {similarity:.3f}
                </div>
                <div class="grid-container">
            """
            
            total_aligned_chars = len(aligned_clean_a) # Should be same as len(aligned_clean_b)
            
            for line_start_idx in range(0, total_aligned_chars, chars_per_line):
                line_end_idx = min(line_start_idx + chars_per_line, total_aligned_chars)
                
                current_line_chars_a = aligned_clean_a[line_start_idx:line_end_idx]
                current_line_chars_b = aligned_clean_b[line_start_idx:line_end_idx]
                
                html += '<div class="row-pair">'
                
                # Row for Document A
                html += '<div class="grid-row">'
                html += f'<div class="row-label">文档A</div>'
                for k in range(len(current_line_chars_a)):
                    char_a_disp = current_line_chars_a[k]
                    char_b_disp = current_line_chars_b[k]
                    
                    status = CharacterGridAligner.get_character_status(char_a_disp, char_b_disp)
                    display_char = char_a_disp if char_a_disp else '□' # Show placeholder for empty
                    css_class = status if char_a_disp else 'empty' # If char_a is empty, it implies B has char (insert) or both empty
                    if char_a_disp == '' and char_b_disp == '': css_class = 'empty' # Should not happen if align_texts is correct
                    elif char_a_disp == '' and char_b_disp != '': css_class = 'empty' # A is empty, B has char (insert for A)
                    elif char_a_disp !='' and status == 'insert': css_class = 'replace' # This case might need review based on status logic

                    html += f'<div class="char-cell {css_class}">{display_char}</div>'
                html += '</div>' # End grid-row for A
                
                # Row for Document B
                html += '<div class="grid-row">'
                html += f'<div class="row-label">文档B</div>'
                for k in range(len(current_line_chars_b)):
                    char_a_disp = current_line_chars_a[k]
                    char_b_disp = current_line_chars_b[k]
                    
                    status = CharacterGridAligner.get_character_status(char_a_disp, char_b_disp)
                    display_char = char_b_disp if char_b_disp else '□'
                    css_class = status if char_b_disp else 'empty'
                    if char_b_disp == '' and char_a_disp == '': css_class = 'empty'
                    elif char_b_disp == '' and char_a_disp != '': css_class = 'empty' # B is empty, A has char (delete for A)
                    
                    html += f'<div class="char-cell {css_class}">{display_char}</div>'
                html += '</div>' # End grid-row for B
                
                html += '</div>' # End row-pair
            
            html += '</div></div>' # End grid-container and segment-pair
        
        html += """
            </div>
        </body>
        </html>
        """
        return html

def main():
    st.title("📚 中文文本对齐工具")
    st.markdown("*基于断句标点的古籍文献平行文本对齐分析工具*")
    
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'alignment_table' not in st.session_state:
        st.session_state.alignment_table = None
    
    st.sidebar.header("参数设置")
    
    uploaded_files = st.sidebar.file_uploader(
        "上传中文文档 (2个文件)", 
        type=["txt", "docx", "pdf", "md"],
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    # Limit to 2 files
    if uploaded_files and len(uploaded_files) > 2:
        st.sidebar.warning("⚠️ 请上传2个文件进行对比。仅处理前两个文件。")
        uploaded_files = uploaded_files[:2]
    elif uploaded_files and len(uploaded_files) < 2:
        st.sidebar.info("ℹ️ 请上传2个文件以进行对比分析。")


    similarity_threshold = st.sidebar.slider(
        "相似度阈值 (段落匹配)",
        min_value=0.1, # Lowered min for more flexibility
        max_value=0.99,
        value=0.6,
        step=0.01,
        help="段落匹配的最低相似度分数 (TF-IDF Cosine Similarity)"
    )
    
    min_segment_length = st.sidebar.slider(
        "最小段落长度 (有效字符数)",
        min_value=5,
        max_value=50,
        value=10,
        step=1,
        help="段落分割后，每个段落包含的最小有效字符数（不计标点）"
    )
    
    chars_per_line = st.sidebar.slider(
        "网格每行字符数 (字符网格视图)",
        min_value=10,
        max_value=80, # Increased max
        value=30,
        step=1,
        help="字符网格视图每行显示的字符数量"
    )
    
    process_button = st.sidebar.button("🚀 开始对齐分析", type="primary", use_container_width=True)
    
    if uploaded_files and len(uploaded_files) == 2 and process_button:
        with st.spinner("⏳ 正在处理文档，请稍候..."):
            try:
                text_processor = ChineseTextProcessor()
                segmenter = ChineseSegmenter()
                matcher = ChineseSimilarityMatcher()
                
                documents = {}
                doc_ids = ['A', 'B'] # For Document A and Document B
                
                progress_bar_streamlit = st.progress(0, text="正在加载文档...")
                
                for i, file_obj in enumerate(uploaded_files):
                    progress_bar_streamlit.progress(i * 0.2, text=f"正在读取文件: {file_obj.name}...")
                    raw_text = text_processor.extract_text_from_file(file_obj)
                    
                    progress_bar_streamlit.progress(i * 0.2 + 0.1, text=f"正在切分段落: {file_obj.name}...")
                    # Normalization is done inside segment_text
                    segments = segmenter.segment_text(raw_text, min_segment_length)
                    
                    doc_id_current = doc_ids[i]
                    segments_with_ids = segmenter.add_segment_ids(segments, doc_id_current)
                    
                    documents[doc_id_current] = {
                        'filename': file_obj.name,
                        'segments': segments_with_ids,
                        'raw_text_preview': raw_text[:500] + ("..." if len(raw_text) > 500 else ""), # For preview
                        'cleaned_text_preview': text_processor.normalize_text(raw_text)[:500] + "..."
                    }
                    st.sidebar.success(f"✓ 文件 {file_obj.name} 处理完毕 ({len(segments)} 段)")
                
                progress_bar_streamlit.progress(0.5, text="正在计算文本相似度...")
                alignment_table = matcher.find_best_matches(
                    documents[doc_ids[0]]['segments'],
                    documents[doc_ids[1]]['segments'],
                    similarity_threshold
                )
                
                progress_bar_streamlit.progress(1.0, text="处理完成!")
                
                st.session_state.processed_data = documents
                st.session_state.alignment_table = alignment_table
                
                matched_count = sum(1 for item in alignment_table if item.get('matched', False))
                st.success(f"🎉 分析完成！找到 {matched_count} 对相似度高于 {similarity_threshold} 的段落。")
                progress_bar_streamlit.empty() # Remove progress bar
                
            except ValueError as ve: # Specific error for file type
                st.error(f"文件处理错误: {str(ve)}")
                if 'progress_bar_streamlit' in locals(): progress_bar_streamlit.empty()
            except Exception as e:
                st.error(f"❌ 处理过程中发生意外错误: {str(e)}")
                st.exception(e) # Shows full traceback for debugging
                if 'progress_bar_streamlit' in locals(): progress_bar_streamlit.empty()
    
    if st.session_state.alignment_table is not None: # Check for None specifically
        alignment_table = st.session_state.alignment_table
        documents = st.session_state.processed_data
        doc_keys = list(documents.keys()) # Should be ['A', 'B']
        
        matched_count = sum(1 for item in alignment_table if item.get('matched', False))
        total_segments_a = len(documents[doc_keys[0]]['segments']) if doc_keys[0] in documents else 0
        total_segments_b = len(documents[doc_keys[1]]['segments']) if doc_keys[1] in documents else 0
        
        st.markdown("---")
        st.subheader("📊 分析结果概览")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("匹配段落对", matched_count)
        with col2:
            if matched_count > 0:
                avg_similarity = np.mean([item['similarity'] for item in alignment_table if item.get('matched', False)])
                st.metric("平均相似度", f"{avg_similarity:.3f}")
            else:
                st.metric("平均相似度", "N/A")
        with col3:
            st.metric(f"A段落总数", total_segments_a)
        with col4:
            st.metric(f"B段落总数", total_segments_b)
        
        tab1, tab2, tab3 = st.tabs(["📜 对齐表格视图", "🔍 字符网格对比", "📈 统计图表"])
        
        with tab1:
            st.markdown("#### 段落对齐详细表")
            st.info("下表显示所有段落。相似度高于阈值的段落已配对在同一行，未配对段落单独列出。内容预览限制为前100字符。")
            
            df_data = []
            # Sort alignment table to group A-only, then matched, then B-only (or by A index then B index)
            # A more intuitive sort: primarily by A's index, then B's if A is missing.
            sorted_alignment = sorted(alignment_table, key=lambda x: (x['a_index'] if x['a_index'] != -1 else float('inf'), x['b_index'] if x['b_index'] != -1 else float('inf')))


            for item in sorted_alignment:
                row = {}
                if item['segment_a']:
                    row['A段落ID'] = item['segment_a']['id']
                    row['A段落内容'] = item['segment_a']['text'][:100] + ('...' if len(item['segment_a']['text']) > 100 else '')
                else:
                    row['A段落ID'] = ''
                    row['A段落内容'] = ''
                
                if item['segment_b']:
                    row['B段落ID'] = item['segment_b']['id']
                    row['B段落内容'] = item['segment_b']['text'][:100] + ('...' if len(item['segment_b']['text']) > 100 else '')
                else:
                    row['B段落ID'] = ''
                    row['B段落内容'] = ''
                
                row['相似度'] = f"{item['similarity']:.3f}" if item['matched'] else "-" # Show similarity only if matched
                df_data.append(row)
            
            df_display = pd.DataFrame(df_data)
            st.dataframe(df_display, use_container_width=True, height=600)
            
            csv_export = df_display.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 下载对齐表格 (CSV)",
                data=csv_export,
                file_name="text_alignment_details.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with tab2:
            st.markdown("#### 字符级别网格对比")
            st.info("高亮显示配对段落间的逐字符差异。仅显示成功匹配的段落对。")
            
            if matched_count > 0:
                grid_aligner = CharacterGridAligner()
                # Filter for only matched pairs to pass to grid HTML generation
                matched_pairs_for_grid = [item for item in alignment_table if item.get('matched', False) and item['segment_a'] and item['segment_b']]
                
                if matched_pairs_for_grid:
                    grid_html_content = grid_aligner.create_grid_html(matched_pairs_for_grid, chars_per_line)
                    st.components.v1.html(grid_html_content, height=800, scrolling=True)
                    
                    st.download_button(
                        label="📥 下载字符网格 (HTML)",
                        data=grid_html_content,
                        file_name="character_grid_comparison.html",
                        mime="text/html",
                        use_container_width=True
                    )
                else: # Should not happen if matched_count > 0, but as a safeguard
                    st.warning("✔️ 有匹配段落，但未能生成网格视图。请检查数据。")
            else:
                st.warning("⚠️ 没有找到符合当前相似度阈值的匹配段落对。请尝试调整侧边栏的“相似度阈值”。")
        
        with tab3:
            st.markdown("#### 数据统计与分析")
            
            if matched_count > 0 and total_segments_a > 0 and total_segments_b > 0 :
                similarities = [item['similarity'] for item in alignment_table if item.get('matched', False)]
                
                plot_col1, plot_col2 = st.columns(2)
                
                with plot_col1:
                    fig_hist = px.histogram(
                        x=similarities,
                        nbins=min(20, len(set(similarities))), # Adjust nbins
                        title="匹配段落相似度分布",
                        labels={'x': '相似度分数', 'y': '段落对数量'},
                        color_discrete_sequence=['#007bff']
                    )
                    fig_hist.update_layout(bargap=0.1, height=400)
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with plot_col2:
                    # Segment lengths (only for matched segments)
                    lengths_a_matched = [len(ChineseTextProcessor.remove_punctuation(item['segment_a']['text'])) for item in alignment_table if item.get('matched', False)]
                    lengths_b_matched = [len(ChineseTextProcessor.remove_punctuation(item['segment_b']['text'])) for item in alignment_table if item.get('matched', False)]
                    
                    fig_scatter = px.scatter(
                        x=lengths_a_matched,
                        y=lengths_b_matched,
                        title="匹配段落长度对比 (有效字符数)",
                        labels={'x': 'A段落长度', 'y': 'B段落长度'},
                        color=similarities, # Color points by similarity
                        color_continuous_scale=px.colors.sequential.Viridis,
                    )
                    fig_scatter.update_layout(height=400)
                    st.plotly_chart(fig_scatter, use_container_width=True)
                
                st.markdown("##### 详细统计数据")
                match_rate_a = (matched_count / total_segments_a * 100) if total_segments_a > 0 else 0
                match_rate_b = (matched_count / total_segments_b * 100) if total_segments_b > 0 else 0
                
                stats_data_dict = {
                    '指标': ['总段落数 (文档A)', '总段落数 (文档B)', '成功匹配段落对数量', 
                             '匹配率 (文档A)', '匹配率 (文档B)', 
                             '平均相似度 (匹配对)', '最高相似度 (匹配对)', '最低相似度 (匹配对)'],
                    '数值': [
                        total_segments_a,
                        total_segments_b,
                        matched_count,
                        f"{match_rate_a:.1f}%",
                        f"{match_rate_b:.1f}%",
                        f"{np.mean(similarities):.3f}" if similarities else "N/A",
                        f"{np.max(similarities):.3f}" if similarities else "N/A",
                        f"{np.min(similarities):.3f}" if similarities else "N/A"
                    ]
                }
                stats_df_display = pd.DataFrame(stats_data_dict)
                st.dataframe(stats_df_display, use_container_width=True, hide_index=True)
            else:
                st.warning("⚠️ 无匹配段落或段落数据不足，无法生成详细统计分析。")
    elif process_button and (not uploaded_files or len(uploaded_files) != 2):
         st.warning("👈 请先在侧边栏上传两个文件并点击“开始对齐分析”。")


if __name__ == "__main__":
    main()