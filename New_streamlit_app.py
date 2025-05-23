import streamlit as st
import pandas as pd
import numpy as np
import re
import io # Not directly used, but common.
import difflib
from typing import List, Tuple, Dict, Optional
import unicodedata
# import base64 # Not used
# from pathlib import Path # Not used

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
# from plotly.subplots import make_subplots # Not directly used
import plotly.express as px
import logging # For setting jieba log level

# Configure page
st.set_page_config(
    page_title="Chinese Text Alignment Tool",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Suppress verbose output from jieba
jieba.setLogLevel(logging.INFO)

MAX_DOCS = 5
DOC_LABELS_AVAILABLE = [chr(65 + i) for i in range(MAX_DOCS)] # ['A', 'B', 'C', 'D', 'E']

class ChineseTextProcessor:
    @staticmethod
    def normalize_text(text: str) -> str:
        text = unicodedata.normalize('NFC', text)
        text = re.sub(r'[\u3000\uFEFF\u200B-\u200D\u2060]', '', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n[ \t]*\n', '\n\n', text)
        return text.strip()

    @staticmethod
    def remove_punctuation(text: str, keep_space=False, keep_newline=False) -> str:
        # Define a base pattern for punctuation to remove
        # Chinese and common English punctuation
        punctuation_to_remove = r',.!?;:()\[\]{}\'"、，。！？；：（）【】《》""''\\-—_'
        
        # Conditionally add space and newline to the removal pattern
        pattern_list = [re.escape(p) for p in punctuation_to_remove]

        if not keep_space:
            pattern_list.append(r'\s') # This will catch spaces and newlines if not kept
        elif not keep_newline: # Space is kept, but newline might not be
            pattern_list.append(r'\n')


        # If keep_space is True AND keep_newline is True, then \s and \n are NOT added to pattern_list
        # meaning they are preserved.
        # If keep_space is False, \s is added, which also removes \n.
        # If keep_space is True but keep_newline is False, \n is added.

        full_pattern = r'[' + ''.join(pattern_list) + r']'
        if not pattern_list: # If somehow all are kept, return original (should not happen with default punc)
            return text
            
        text = re.sub(full_pattern, '', text)
        return text

    @staticmethod
    def extract_text_from_file(uploaded_file) -> str:
        file_type = uploaded_file.name.split('.')[-1].lower()
        try:
            if file_type == 'txt':
                try:
                    return str(uploaded_file.read(), 'utf-8')
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    return str(uploaded_file.read(), 'gbk', errors='replace')
            elif file_type == 'docx':
                doc = docx.Document(uploaded_file)
                return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
            elif file_type == 'pdf':
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                text = ''
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
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
        except Exception as e:
            st.error(f"Error reading file {uploaded_file.name} (type: {file_type}): {e}")
            return "" # Return empty string on error

class ChineseSegmenter:
    def segment_text(self, text: str, min_length: int = 10, delimiters: List[str] = None) -> List[str]:
        if delimiters is None or not delimiters: # If no delimiters selected, treat as one segment or handle very long text
            delimiters = ['。', '？', '!', '；', '\n'] # Default fallback
            if not delimiters: # Should not happen with fallback, but safety for empty list
                processed_text_content = ChineseTextProcessor.remove_punctuation(text, keep_space=True, keep_newline=True).strip()
                if len(processed_text_content) >= min_length:
                    return [text.strip()] if text.strip() else []
                return []


        normalized_text = ChineseTextProcessor.normalize_text(text)

        # Escape delimiters for regex and ensure they are treated as alternatives
        # The logic for re.split with capturing groups:
        # result will be [text_before_delim1, delim1, text_between_delim1_and_2, delim2, ...]
        escaped_delimiters = [re.escape(d) for d in delimiters if d] # Filter out empty delimiters
        if not escaped_delimiters: # If all delimiters were empty strings
            processed_text_content = ChineseTextProcessor.remove_punctuation(normalized_text, keep_space=True, keep_newline=True).strip()
            if len(processed_text_content) >= min_length:
                return [normalized_text.strip()] if normalized_text.strip() else []
            return []

        split_pattern_str = r'(' + '|'.join(escaped_delimiters) + r')'
        parts = re.split(split_pattern_str, normalized_text)

        segments = []
        current_segment_text = ""

        for i, part_content in enumerate(parts):
            if not part_content: # Skip empty strings that re.split might produce
                continue
            
            is_delimiter_part = (i % 2 == 1) # In re.split with capturing group, odd indices are delimiters

            current_segment_text += part_content # Append current part (text or delimiter)

            # Condition to finalize a segment:
            # 1. If the part just added was a delimiter (and not just a space if space is a delimiter)
            # 2. And the accumulated current_segment_text (content part) meets min_length
            if is_delimiter_part:
                # Check content length (excluding punctuation, but including spaces/newlines if they are not delimiters themselves)
                content_for_len_check = ChineseTextProcessor.remove_punctuation(current_segment_text, 
                                                                              keep_space=" " not in delimiters, 
                                                                              keep_newline="\n" not in delimiters)
                if len(content_for_len_check.strip()) >= min_length:
                    segments.append(current_segment_text.strip())
                    current_segment_text = ""
        
        # Add any remaining part
        if current_segment_text.strip():
            segments.append(current_segment_text.strip())
            current_segment_text = ""


        # Merge short segments: if a segment is too short, merge with the next one.
        if not segments: return []

        merged_segments = []
        idx = 0
        while idx < len(segments):
            current_s = segments[idx].strip()
            if not current_s:
                idx += 1
                continue
            
            # Check length of text content for "shortness"
            # Important: for length check, remove all punctuation specified by the user,
            # but keep spaces/newlines if they are NOT delimiters themselves for a more accurate "content" length.
            content_for_len_check = ChineseTextProcessor.remove_punctuation(current_s,
                                                                          keep_space=" " not in delimiters,
                                                                          keep_newline="\n" not in delimiters)

            if len(content_for_len_check.strip()) < min_length:
                if idx + 1 < len(segments): # If there's a next segment
                    next_s = segments[idx+1].strip()
                    # Merge: if current ends with space/newline or next starts with it, simple concat. Else add space.
                    if current_s.endswith(tuple(delimiters + [' '])) or \
                       next_s.startswith(tuple(delimiters + [' '])) or \
                       "\n" in current_s or "\n" in next_s : # A bit heuristic for spacing
                        merged_s = current_s + next_s
                    else:
                        merged_s = current_s + " " + next_s # Default to adding a space
                    
                    segments[idx+1] = merged_s # Replace next segment with merged
                    # Do not add current_s to merged_segments yet. The merged segment will be processed in next iteration.
                else: # It's the last segment and it's too short
                    if merged_segments: # Append to the previously added segment if one exists
                        # Similar spacing heuristic
                        if merged_segments[-1].endswith(tuple(delimiters + [' '])) or \
                           current_s.startswith(tuple(delimiters + [' '])) or \
                           "\n" in merged_segments[-1] or "\n" in current_s:
                            merged_segments[-1] += current_s
                        else:
                            merged_segments[-1] += " " + current_s
                    else: # It's the only segment and it's short
                        merged_segments.append(current_s)
            else: # Segment is long enough
                merged_segments.append(current_s)
            idx += 1
            
        return [s.strip() for s in merged_segments if s.strip()]


    def add_segment_ids(self, segments: List[str], doc_id_prefix: str) -> List[Dict]:
        return [
            {
                'id': f"{doc_id_prefix}{(i+1):03d}",
                'text': segment,
                'doc_id_prefix': doc_id_prefix, # Store the prefix (A, B, etc.)
                'index': i
            }
            for i, segment in enumerate(segments)
        ]

class ChineseSimilarityMatcher:
    def __init__(self):
        def chinese_analyzer(text):
            return jieba.lcut(text)
        self.vectorizer = TfidfVectorizer(analyzer=chinese_analyzer, min_df=1)

    def find_best_matches(self,
                          segments_data_A: List[Dict],
                          segments_data_B: List[Dict],
                          threshold: float = 0.6) -> List[Dict]:
        if not segments_data_A or not segments_data_B:
            alignment_table = []
            for i, seg_a_data in enumerate(segments_data_A):
                alignment_table.append({'segment_a': seg_a_data, 'segment_b': None, 'similarity': 0.0, 'a_index': i, 'b_index': -1, 'matched': False})
            for j, seg_b_data in enumerate(segments_data_B):
                alignment_table.append({'segment_a': None, 'segment_b': seg_b_data, 'similarity': 0.0, 'a_index': -1, 'b_index': j, 'matched': False})
            return alignment_table

        texts_A = [s['text'] for s in segments_data_A]
        texts_B = [s['text'] for s in segments_data_B]

        clean_texts_A = [ChineseTextProcessor.remove_punctuation(t, keep_space=False, keep_newline=False) for t in texts_A]
        clean_texts_B = [ChineseTextProcessor.remove_punctuation(t, keep_space=False, keep_newline=False) for t in texts_B]
        
        # Filter out potentially empty strings after punctuation removal for TF-IDF
        # Store original indices to map back if needed, though here we operate on full lists
        valid_clean_A = [t for t in clean_texts_A if t.strip()]
        valid_clean_B = [t for t in clean_texts_B if t.strip()]

        if not valid_clean_A or not valid_clean_B: # If one list becomes all empty
            similarity_matrix = np.zeros((len(texts_A), len(texts_B)))
        else:
            all_valid_clean_texts = valid_clean_A + valid_clean_B
            self.vectorizer.fit(all_valid_clean_texts)
            
            # Transform original clean_texts lists (they might contain empty strings where valid_clean_* do not)
            vectors_A = self.vectorizer.transform(clean_texts_A).toarray()
            vectors_B = self.vectorizer.transform(clean_texts_B).toarray()
            
            if vectors_A.shape[1] == 0 or vectors_B.shape[1] == 0 or vectors_A.size == 0 or vectors_B.size == 0:
                similarity_matrix = np.zeros((len(texts_A), len(texts_B)))
            else:
                try:
                    similarity_matrix = cosine_similarity(vectors_A, vectors_B)
                except ValueError:
                    similarity_matrix = np.zeros((len(texts_A), len(texts_B)))
        
        alignment_table = []
        used_b_indices = set()

        for i, seg_a_data in enumerate(segments_data_A):
            best_score = -1.0
            best_j = -1
            if i < similarity_matrix.shape[0]:
                for j in range(len(segments_data_B)):
                    if j in used_b_indices:
                        continue
                    if j < similarity_matrix.shape[1]:
                        score = similarity_matrix[i, j]
                        if score >= threshold and score > best_score:
                            best_score = score
                            best_j = j
            
            if best_j != -1:
                alignment_table.append({
                    'segment_a': seg_a_data, 'segment_b': segments_data_B[best_j], 'similarity': best_score,
                    'a_index': i, 'b_index': best_j, 'matched': True
                })
                used_b_indices.add(best_j)
            else:
                alignment_table.append({
                    'segment_a': seg_a_data, 'segment_b': None, 'similarity': 0.0,
                    'a_index': i, 'b_index': -1, 'matched': False
                })
        
        for j, seg_b_data in enumerate(segments_data_B):
            if j not in used_b_indices:
                alignment_table.append({
                    'segment_a': None, 'segment_b': seg_b_data, 'similarity': 0.0,
                    'a_index': -1, 'b_index': j, 'matched': False
                })
        return alignment_table

class CharacterGridAligner:
    @staticmethod
    def align_full_texts_for_grid(text1_full: str, text2_full: str, ignore_punctuation_for_alignment_and_display: bool) -> Tuple[List[str], List[str]]:
        if ignore_punctuation_for_alignment_and_display:
            processed_text1 = ChineseTextProcessor.remove_punctuation(text1_full, keep_space=False, keep_newline=False)
            processed_text2 = ChineseTextProcessor.remove_punctuation(text2_full, keep_space=False, keep_newline=False)
            # Display characters will be from these processed texts
            display_source1_list = list(processed_text1)
            display_source2_list = list(processed_text2)
        else:
            processed_text1 = text1_full # Align on original text with punctuation
            processed_text2 = text2_full
            # Display characters will be from original texts
            display_source1_list = list(text1_full)
            display_source2_list = list(text2_full)

        if not processed_text1 and not processed_text2:
            return [], [] # Both effectively empty

        matcher = difflib.SequenceMatcher(None, processed_text1, processed_text2, autojunk=False)
        opcodes = matcher.get_opcodes()

        aligned_display_chars1 = []
        aligned_display_chars2 = []

        for tag, i1, i2, j1, j2 in opcodes:
            # i1, i2 refer to processed_text1; j1, j2 refer to processed_text2
            # We use these indices to slice from display_sourceX_list
            if tag == 'equal':
                aligned_display_chars1.extend(display_source1_list[i1:i2])
                aligned_display_chars2.extend(display_source2_list[j1:j2])
            elif tag == 'delete': # In text1 but not text2
                aligned_display_chars1.extend(display_source1_list[i1:i2])
                aligned_display_chars2.extend([''] * (i2 - i1)) # Placeholders for text2
            elif tag == 'insert': # In text2 but not text1
                aligned_display_chars1.extend([''] * (j2 - j1)) # Placeholders for text1
                aligned_display_chars2.extend(display_source2_list[j1:j2])
            elif tag == 'replace':
                len_op_text1 = i2 - i1
                len_op_text2 = j2 - j1
                
                op_segment1_display = display_source1_list[i1:i2]
                op_segment2_display = display_source2_list[j1:j2]

                for k_replace in range(max(len_op_text1, len_op_text2)):
                    aligned_display_chars1.append(op_segment1_display[k_replace] if k_replace < len_op_text1 else '')
                    aligned_display_chars2.append(op_segment2_display[k_replace] if k_replace < len_op_text2 else '')
        
        return aligned_display_chars1, aligned_display_chars2

    @staticmethod
    def get_character_status(char1_display: str, char2_display: str) -> str:
        if char1_display == char2_display and char1_display != '':
            return 'equal'
        elif char1_display == '' and char2_display != '':
            return 'insert' # Content in Doc B, placeholder in Doc A
        elif char2_display == '' and char1_display != '':
            return 'delete' # Content in Doc A, placeholder in Doc B
        elif char1_display != '' and char2_display != '' and char1_display != char2_display:
            return 'replace'
        else: # Both are '', or one is '' and status couldn't be determined above (should be rare)
            return 'empty'

    @staticmethod
    def create_grid_html_for_full_docs(
            doc_label_A: str, doc_label_B: str,
            aligned_chars_A: List[str],
            aligned_chars_B: List[str],
            chars_per_line: int
        ) -> str:
        html_start = f"""
        <!DOCTYPE html><html lang="zh"><head><meta charset="UTF-8"><title>文档字符对齐网格</title>
        <style>
            body {{ font-family: "Source Han Serif SC", "Noto Serif CJK SC", "SimSun", serif; background: #f0f2f5; margin: 0; padding: 15px; line-height: 1.1; }}
            .container {{ max-width: 98%; margin: 0 auto; background: white; border-radius: 6px; padding: 15px; box-shadow: 0 1px 6px rgba(0,0,0,0.08); }}
            h1 {{ text-align: center; color: #2c3e50; margin-bottom: 15px; font-size: 1.5em; }}
            .grid-docs-header {{ text-align: center; font-weight: bold; margin-bottom: 12px; font-size: 1.05em; color: #34495e; }}
            .grid-container {{ padding: 8px; background: #fdfdfd; border: 1px solid #e0e0e0; border-radius: 4px; }}
            .row-pair {{ margin-bottom: 3px; }} /* Compact row pairs */
            .grid-row {{ display: flex; margin-bottom: 0px; align-items: stretch; }} /* No margin between A and B rows */
            .row-label {{ min-width: 55px; width: auto; padding: 0 4px; font-weight: 600; color: #4a5568; font-size: 0.7em; text-align: center; margin-right: 4px; display: flex; align-items: center; justify-content: center; background: #e9ecef; border-radius: 2px; }}
            .char-cell {{ width: 28px; height: 28px; display: flex; align-items: center; justify-content: center; border: 1px solid #d1d5db; font-size: 14px; margin-right: 1px; box-sizing: border-box; font-family: "SimSun", "MingLiU", monospace; overflow: hidden; }}
            .equal {{ background: #e6ffed; color: #1e4620; border-color: #b8e0bb; }}
            .insert {{ background: #ffebe6; color: #6b2211; border-color: #f5b8a9; }} /* Doc B unique */
            .delete {{ background: #fff9e6; color: #735c10; border-color: #f7e8aa; }} /* Doc A unique */
            .replace {{ background: #efedff; color: #3a328f; border-color: #c8c2f2; }}
            .empty {{ background: #f8f9fa; border-color: #dee2e6; color: #adb5bd; }}
            .legend {{ display: flex; justify-content: center; gap: 10px; margin: 12px 0; flex-wrap: wrap; }}
            .legend-item {{ display: flex; align-items: center; gap: 4px; font-size: 0.75em; }}
            .legend-color {{ width: 14px; height: 14px; border-radius: 2px; border: 1px solid #b0b0b0; }}
        </style></head><body><div class="container">
        <h1>文档字符对齐网格视图</h1>
        <div class="grid-docs-header">文档 {doc_label_A} vs 文档 {doc_label_B}</div>
        <div class="legend">
            <div class="legend-item"><div class="legend-color equal"></div><span>相同</span></div>
            <div class="legend-item"><div class="legend-color insert"></div><span>文档 {doc_label_B} 独有</span></div>
            <div class="legend-item"><div class="legend-color delete"></div><span>文档 {doc_label_A} 独有</span></div>
            <div class="legend-item"><div class="legend-color replace"></div><span>不同</span></div>
            <div class="legend-item"><div class="legend-color empty"></div><span>空位</span></div>
        </div>
        <div class="grid-container">
        """
        html_content_main = ""
        total_aligned_length = len(aligned_chars_A) # Should be same as aligned_chars_B
        ptr = 0
        while ptr < total_aligned_length:
            line_end_ptr = min(ptr + chars_per_line, total_aligned_length)
            
            current_block_A_chars = aligned_chars_A[ptr:line_end_ptr]
            current_block_B_chars = aligned_chars_B[ptr:line_end_ptr]

            # Determine if a row should be skipped for one of the documents
            # A is all content, B is all placeholders for this block
            is_A_solid_B_empty_block = all(c_a != '' for c_a in current_block_A_chars) and \
                                       all(c_b == '' for c_b in current_block_B_chars)
            # B is all content, A is all placeholders for this block
            is_B_solid_A_empty_block = all(c_b != '' for c_b in current_block_B_chars) and \
                                       all(c_a == '' for c_a in current_block_A_chars)
            
            html_content_main += '<div class="row-pair">'
            # --- Document A Row ---
            if not is_B_solid_A_empty_block: # Don't show A's row if it's all empty and B is solid
                html_content_main += f'<div class="grid-row"><div class="row-label">{doc_label_A}</div>'
                for k in range(len(current_block_A_chars)):
                    char_a = current_block_A_chars[k]
                    char_b = current_block_B_chars[k]
                    status = CharacterGridAligner.get_character_status(char_a, char_b)
                    
                    display_char = char_a if char_a else ('□' if not is_A_solid_B_empty_block else '') # Show '□' if B has content or mixed
                    css_class = status if char_a else 'empty'
                    if char_a == '' and is_A_solid_B_empty_block : css_class = 'empty' # Should not happen

                    html_content_main += f'<div class="char-cell {css_class}">{display_char}</div>'
                html_content_main += '</div>'

            # --- Document B Row ---
            if not is_A_solid_B_empty_block: # Don't show B's row if it's all empty and A is solid
                html_content_main += f'<div class="grid-row"><div class="row-label">{doc_label_B}</div>'
                for k in range(len(current_block_B_chars)):
                    char_a = current_block_A_chars[k]
                    char_b = current_block_B_chars[k]
                    status = CharacterGridAligner.get_character_status(char_a, char_b)
                    
                    display_char = char_b if char_b else ('□' if not is_B_solid_A_empty_block else '') # Show '□' if A has content or mixed
                    css_class = status if char_b else 'empty'
                    if char_b == '' and is_B_solid_A_empty_block : css_class = 'empty' # Should not happen

                    html_content_main += f'<div class="char-cell {css_class}">{display_char}</div>'
                html_content_main += '</div>'
            
            html_content_main += '</div>' # End row-pair
            ptr = line_end_ptr
        
        html_end = "</div></div></div></body></html>"
        return html_start + html_content_main + html_end


def initialize_session_state():
    if 'processed_documents_data' not in st.session_state: # Stores dict of {doc_label: {data}}
        st.session_state.processed_documents_data = {}
    if 'uploaded_file_objects_info' not in st.session_state: # Stores list of (file_object, assigned_label, original_name)
        st.session_state.uploaded_file_objects_info = []
    if 'alignment_table_cache' not in st.session_state: # Cache for pairwise alignment results
        st.session_state.alignment_table_cache = {}
    if 'selected_delimiters_names' not in st.session_state:
        st.session_state.selected_delimiters_names = ["句号 (。)", "问号 (？)", "感叹号 (!)", "分号 (；)", "换行符 (\\n)"]
    if 'active_delimiters_chars' not in st.session_state:
        st.session_state.active_delimiters_chars = ['。', '？', '!', '；', '\n']


ALL_POSSIBLE_DELIMITERS = {
    "句号 (。)": "。", "问号 (？)": "？", "感叹号 (!)": "!", "分号 (；)": "；",
    "逗号 (，)": "，", "顿号 (、)": "、", "冒号 (：)": "：",
    "换行符 (↵)": "\n", "空格 (␣)": " "
}

def main():
    st.title("📚 CTextAlign 中文文本对齐工具")
    st.markdown("*支持多文档、自定义分词、高级字符网格比对*")

    initialize_session_state()

    # --- Sidebar: File Management ---
    st.sidebar.header("📄 文档管理")
    newly_uploaded_streamlit_files = st.sidebar.file_uploader(
        f"上传中文文档 (最多 {MAX_DOCS} 个)",
        type=["txt", "docx", "pdf", "md"],
        accept_multiple_files=True,
        key=f"file_uploader_main_{len(st.session_state.uploaded_file_objects_info)}" # Dynamic key
    )

    if newly_uploaded_streamlit_files:
        for up_file in newly_uploaded_streamlit_files:
            if len(st.session_state.uploaded_file_objects_info) < MAX_DOCS:
                if not any(info[2] == up_file.name for info in st.session_state.uploaded_file_objects_info):
                    current_labels_in_use = {info[1] for info in st.session_state.uploaded_file_objects_info}
                    assigned_label = next((l for l in DOC_LABELS_AVAILABLE if l not in current_labels_in_use), None)
                    if assigned_label:
                        st.session_state.uploaded_file_objects_info.append((up_file, assigned_label, up_file.name))
                    else: st.sidebar.warning("无法分配文档标签。")
                else: st.sidebar.warning(f"文件 '{up_file.name}' 已存在。")
            else: st.sidebar.warning(f"已达到最大文档数 ({MAX_DOCS})。"); break
        st.rerun()

    if st.session_state.uploaded_file_objects_info:
        st.sidebar.subheader("已上传文档列表:")
        
        # Make a mutable copy for iteration if needed, or directly modify session state if careful
        # We iterate and potentially modify st.session_state.uploaded_file_objects_info directly
        
        for i in range(len(st.session_state.uploaded_file_objects_info)): # Iterate with index
            # It's safer to fetch fresh from session state if reruns happen inside the loop,
            # but for this selectbox logic, direct modification is okay if rerun is only after a change.
            file_obj, current_assigned_label, original_name = st.session_state.uploaded_file_objects_info[i]
            
            cols = st.sidebar.columns([4, 3, 1])
            cols[0].markdown(f"**{original_name[:20]}{'...' if len(original_name)>20 else ''}**")
            
            # --- Improved Label Assignment Logic ---
            num_uploaded_docs = len(st.session_state.uploaded_file_objects_info)
            possible_labels_for_dropdown = DOC_LABELS_AVAILABLE[:num_uploaded_docs] # e.g., ['A', 'B'] if 2 docs

            try:
                current_label_idx_in_dropdown = possible_labels_for_dropdown.index(current_assigned_label)
            except ValueError:
                # This case implies current_assigned_label is somehow not in DOC_LABELS_AVAILABLE up to num_uploaded_docs
                # This could happen if DOC_LABELS_AVAILABLE was shortened (e.g., a doc was deleted)
                # and this doc still has an "older" higher label.
                # Fallback: add it and re-sort, or default to first option.
                # For simplicity, let's ensure it's there.
                if current_assigned_label not in possible_labels_for_dropdown:
                    possible_labels_for_dropdown.append(current_assigned_label)
                    possible_labels_for_dropdown.sort()
                current_label_idx_in_dropdown = possible_labels_for_dropdown.index(current_assigned_label)


            new_label_selected = cols[1].selectbox(
                f"L##{original_name}", # Unique key for each selectbox
                options=possible_labels_for_dropdown,
                index=current_label_idx_in_dropdown,
                label_visibility="collapsed"
            )

            if new_label_selected != current_assigned_label:
                # Find if another document currently has the new_label_selected
                other_doc_index_with_new_label = -1
                for k, (_, other_label, _) in enumerate(st.session_state.uploaded_file_objects_info):
                    if k != i and other_label == new_label_selected:
                        other_doc_index_with_new_label = k
                        break
                
                # Update current document's label
                st.session_state.uploaded_file_objects_info[i] = (
                    st.session_state.uploaded_file_objects_info[i][0], # file_obj
                    new_label_selected,
                    st.session_state.uploaded_file_objects_info[i][2]  # original_name
                )

                # If another document had the new_label_selected, swap its label to current_assigned_label (old label of current doc)
                if other_doc_index_with_new_label != -1:
                    st.session_state.uploaded_file_objects_info[other_doc_index_with_new_label] = (
                        st.session_state.uploaded_file_objects_info[other_doc_index_with_new_label][0], # file_obj
                        current_assigned_label, # Give it the old label of the doc we just changed
                        st.session_state.uploaded_file_objects_info[other_doc_index_with_new_label][2]  # original_name
                    )
                
                # After changing labels, re-sort the list by label to maintain A, B, C... order visually (optional but good UX)
                st.session_state.uploaded_file_objects_info.sort(key=lambda x: x[1])
                st.rerun() # Rerun to reflect changes in UI and ensure consistency

            # --- End of Improved Label Assignment Logic ---

            if cols[2].button("🗑️", key=f"del_{original_name}", help=f"删除 {original_name}"):
                # Store the label of the document being deleted
                deleted_label = st.session_state.uploaded_file_objects_info[i][1]
                
                st.session_state.uploaded_file_objects_info.pop(i) # Remove the document
                
                if original_name in st.session_state.processed_documents_data:
                    del st.session_state.processed_documents_data[original_name]
                
                keys_to_del_from_cache = [k_cache for k_cache in st.session_state.alignment_table_cache if original_name in k_cache]
                for k_del in keys_to_del_from_cache:
                    del st.session_state.alignment_table_cache[k_del]

                # Re-assign labels if necessary to keep them contiguous (A, B, C...) after deletion
                # This is important if a doc with label 'B' is deleted when 'C' exists. 'C' should become 'B'.
                current_labels_after_delete = {info[1] for info in st.session_state.uploaded_file_objects_info}
                expected_labels = DOC_LABELS_AVAILABLE[:len(st.session_state.uploaded_file_objects_info)]
                
                if set(current_labels_after_delete) != set(expected_labels):
                    # Simple re-labeling: sort by original upload order (implicit by current list order)
                    # or by filename, then assign A, B, C...
                    # For now, let's sort by filename then re-label, as original upload order isn't stored robustly.
                    st.session_state.uploaded_file_objects_info.sort(key=lambda x: x[2]) # Sort by original_name
                    for k_relabel in range(len(st.session_state.uploaded_file_objects_info)):
                        st.session_state.uploaded_file_objects_info[k_relabel] = (
                            st.session_state.uploaded_file_objects_info[k_relabel][0],
                            DOC_LABELS_AVAILABLE[k_relabel], # Assign A, B, C...
                            st.session_state.uploaded_file_objects_info[k_relabel][2]
                        )
                
                st.rerun()
                return # Crucial: stop script execution for this iteration after deletion and rerun
        
        # This sort should happen outside the loop if a rerun didn't occur inside the loop due to label change
        # However, if a label change occurs, a rerun happens, and this line might not be strictly necessary here.
        # Keeping it for safety to ensure order after all selectboxes are processed in a run without changes.
        # st.session_state.uploaded_file_objects_info.sort(key=lambda x: x[1]) # Sort by assigned label (A, B, C...)

    # --- Sidebar: Processing Parameters ---
    st.sidebar.header("⚙️ 处理参数")
    min_segment_len = st.sidebar.slider("最小段落长度 (有效字符)", 3, 50, 10, 1, help="段落分割后，每个段落最少包含的有效字符数（不计标点）。")
    
    st.sidebar.subheader("段落分割符选择")
    temp_selected_delimiters_names = []
    for name, char_val in ALL_POSSIBLE_DELIMITERS.items():
        if st.sidebar.checkbox(name, value=(name in st.session_state.selected_delimiters_names), key=f"delim_cb_{name}"):
            temp_selected_delimiters_names.append(name)
    
    if temp_selected_delimiters_names != st.session_state.selected_delimiters_names:
        st.session_state.selected_delimiters_names = temp_selected_delimiters_names
        st.session_state.active_delimiters_chars = [ALL_POSSIBLE_DELIMITERS[name] for name in st.session_state.selected_delimiters_names if name in ALL_POSSIBLE_DELIMITERS]
        st.rerun() # Rerun if delimiter selection changed

    similarity_thresh = st.sidebar.slider("相似度阈值 (段落匹配)", 0.1, 0.99, 0.6, 0.01)
    chars_per_line_grid = st.sidebar.slider("网格每行字符数 (字符网格)", 10, 80, 30, 1)

    process_all_button = st.sidebar.button("🚀 处理所有已上传文档", type="primary", use_container_width=True,
                                         disabled=not st.session_state.uploaded_file_objects_info)

    # --- Main Processing Logic ---
    if process_all_button:
        if not st.session_state.uploaded_file_objects_info:
            st.error("请先上传文档后再进行处理。")
        else:
            with st.spinner("⏳ 正在处理所有文档..."):
                text_processor = ChineseTextProcessor()
                segmenter = ChineseSegmenter()
                
                # Clear previous processed data to ensure fresh processing
                st.session_state.processed_documents_data = {}
                st.session_state.alignment_table_cache = {} # Clear pairwise alignment cache

                for file_obj, assigned_label, original_name in st.session_state.uploaded_file_objects_info:
                    st.write(f"处理文档: {original_name} (标签: {assigned_label})...") # Progress update
                    raw_text = text_processor.extract_text_from_file(file_obj)
                    if not raw_text:
                        st.warning(f"文档 {original_name} 内容为空或读取失败。")
                        continue

                    segments = segmenter.segment_text(raw_text, min_segment_len, st.session_state.active_delimiters_chars)
                    segments_with_ids = segmenter.add_segment_ids(segments, assigned_label)
                    
                    st.session_state.processed_documents_data[original_name] = {
                        'doc_label': assigned_label,
                        'filename': original_name,
                        'full_raw_text': raw_text, # Store full raw text
                        'segments_data': segments_with_ids,
                        'segment_count': len(segments_with_ids)
                    }
                st.success("🎉 所有文档处理完成！")
                st.rerun() # Rerun to update UI with processed data

    # --- Display Area ---
    if st.session_state.processed_documents_data:
        processed_doc_infos = list(st.session_state.processed_documents_data.values())
        processed_doc_infos.sort(key=lambda x: x['doc_label']) # Sort by A, B, C...
        
        doc_display_names_map = {info['doc_label']: info['filename'] for info in processed_doc_infos}
        available_doc_labels_for_selection = [info['doc_label'] for info in processed_doc_infos]

        if len(available_doc_labels_for_selection) < 2:
            st.info("请处理至少两个文档以进行对比分析。")
        else:
            st.markdown("---")
            st.subheader("🔬 选择文档进行对比分析")
            col_sel1, col_sel2 = st.columns(2)
            selected_doc_label_1 = col_sel1.selectbox(
                "选择文档 1:", available_doc_labels_for_selection, index=0,
                format_func=lambda x: f"{x} ({doc_display_names_map.get(x, '')})", key="sel_doc_1"
            )
            # Ensure Doc 2 default is different from Doc 1
            doc2_default_idx = 1 if len(available_doc_labels_for_selection) > 1 else 0
            if available_doc_labels_for_selection[doc2_default_idx] == selected_doc_label_1 and len(available_doc_labels_for_selection) > 1:
                 doc2_default_idx = 0 # Should not happen if list > 1 and default is 1. Pick 0 if it's the only other option.

            selected_doc_label_2 = col_sel2.selectbox(
                "选择文档 2:", available_doc_labels_for_selection, index=doc2_default_idx,
                format_func=lambda x: f"{x} ({doc_display_names_map.get(x, '')})", key="sel_doc_2"
            )

            if selected_doc_label_1 == selected_doc_label_2:
                st.warning("请选择两个不同的文档进行对比。")
            else:
                # Find original filenames for these labels to fetch data
                doc_data_1 = next(item for item in processed_doc_infos if item['doc_label'] == selected_doc_label_1)
                doc_data_2 = next(item for item in processed_doc_infos if item['doc_label'] == selected_doc_label_2)

                # --- Alignment Table and Stats (Pairwise) ---
                tab_align, tab_grid, tab_stats = st.tabs(["📜 对齐表格视图", "⠿ 字符网格对比", "📈 统计图表"])

                with tab_align:
                    st.markdown(f"#### 段落对齐: **{doc_data_1['filename']} ({selected_doc_label_1})** vs **{doc_data_2['filename']} ({selected_doc_label_2})**")
                    
                    cache_key_align = tuple(sorted((doc_data_1['filename'], doc_data_2['filename']))) + (similarity_thresh,)
                    if cache_key_align not in st.session_state.alignment_table_cache:
                        with st.spinner(f"计算 {selected_doc_label_1} 和 {selected_doc_label_2} 的段落对齐..."):
                            matcher_align = ChineseSimilarityMatcher()
                            st.session_state.alignment_table_cache[cache_key_align] = matcher_align.find_best_matches(
                                doc_data_1['segments_data'], doc_data_2['segments_data'], similarity_thresh
                            )
                    
                    current_alignment_table = st.session_state.alignment_table_cache[cache_key_align]
                    matched_count_pair = sum(1 for item in current_alignment_table if item.get('matched', False))
                    st.info(f"找到 {matched_count_pair} 对相似段落 (阈值 > {similarity_thresh:.2f})。内容预览限100字符。")

                    df_data_align = []
                    sorted_alignment_for_df = sorted(current_alignment_table, key=lambda x: (x['a_index'] if x['a_index'] != -1 else float('inf'), x['b_index'] if x['b_index'] != -1 else float('inf')))
                    for item in sorted_alignment_for_df:
                        row = {}
                        doc1_label_dynamic = selected_doc_label_1 # More dynamic label
                        doc2_label_dynamic = selected_doc_label_2
                        if item['segment_a']:
                            row[f'{doc1_label_dynamic}段落ID'] = item['segment_a']['id']
                            row[f'{doc1_label_dynamic}段落内容'] = item['segment_a']['text'][:100] + ('...' if len(item['segment_a']['text']) > 100 else '')
                        else:
                            row[f'{doc1_label_dynamic}段落ID'] = ''
                            row[f'{doc1_label_dynamic}段落内容'] = ''
                        
                        if item['segment_b']:
                            row[f'{doc2_label_dynamic}段落ID'] = item['segment_b']['id']
                            row[f'{doc2_label_dynamic}段落内容'] = item['segment_b']['text'][:100] + ('...' if len(item['segment_b']['text']) > 100 else '')
                        else:
                            row[f'{doc2_label_dynamic}段落ID'] = ''
                            row[f'{doc2_label_dynamic}段落内容'] = ''
                        row['相似度'] = f"{item['similarity']:.3f}" if item['matched'] else "-"
                        df_data_align.append(row)
                    
                    df_display_align = pd.DataFrame(df_data_align)
                    st.dataframe(df_display_align, use_container_width=True, height=500)
                    csv_export_align = df_display_align.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(f"📥 下载 {selected_doc_label_1}-{selected_doc_label_2} 对齐表 (CSV)", csv_export_align, f"alignment_{selected_doc_label_1}_{selected_doc_label_2}.csv", "text/csv")

                with tab_grid:
                    st.markdown(f"#### 字符网格: **{doc_data_1['filename']} ({selected_doc_label_1})** vs **{doc_data_2['filename']} ({selected_doc_label_2})**")
                    ignore_punc_grid_display = st.checkbox("比对和显示时忽略所有标点符号", value=False, key="cb_ignore_punc_grid")

                    text_A_for_grid = doc_data_1['full_raw_text']
                    text_B_for_grid = doc_data_2['full_raw_text']

                    with st.spinner("生成字符网格..."):
                        aligned_A_grid_chars, aligned_B_grid_chars = CharacterGridAligner.align_full_texts_for_grid(
                            text_A_for_grid, text_B_for_grid, ignore_punc_grid_display
                        )
                    
                    if aligned_A_grid_chars or aligned_B_grid_chars:
                        grid_html_content = CharacterGridAligner.create_grid_html_for_full_docs(
                            selected_doc_label_1, selected_doc_label_2,
                            aligned_A_grid_chars, aligned_B_grid_chars,
                            chars_per_line_grid
                        )
                        st.components.v1.html(grid_html_content, height=700, scrolling=True)
                        st.download_button(f"📥 下载 {selected_doc_label_1}-{selected_doc_label_2} 网格 (HTML)", grid_html_content, f"grid_{selected_doc_label_1}_{selected_doc_label_2}.html", "text/html")
                    else:
                        st.info("选择的文档内容为空或处理后为空，无法生成字符网格。")
                
                with tab_stats:
                    st.markdown(f"#### 统计分析: **{doc_data_1['filename']} ({selected_doc_label_1})** vs **{doc_data_2['filename']} ({selected_doc_label_2})**")
                    current_alignment_table_for_stats = st.session_state.alignment_table_cache.get(cache_key_align, []) # Use cached
                    
                    if current_alignment_table_for_stats:
                        matched_similarities = [item['similarity'] for item in current_alignment_table_for_stats if item.get('matched', False)]
                        if matched_similarities:
                            plot_col_stats1, plot_col_stats2 = st.columns(2)
                            with plot_col_stats1:
                                fig_hist_stats = px.histogram(x=matched_similarities, nbins=20, title="匹配段落相似度分布", labels={'x': '相似度', 'y': '频次'})
                                st.plotly_chart(fig_hist_stats, use_container_width=True)
                            
                            with plot_col_stats2:
                                lengths_a_matched_stats = [len(ChineseTextProcessor.remove_punctuation(item['segment_a']['text'])) for item in current_alignment_table_for_stats if item.get('matched', False)]
                                lengths_b_matched_stats = [len(ChineseTextProcessor.remove_punctuation(item['segment_b']['text'])) for item in current_alignment_table_for_stats if item.get('matched', False)]
                                fig_scatter_stats = px.scatter(x=lengths_a_matched_stats, y=lengths_b_matched_stats, title="匹配段落长度对比", labels={'x': f'{selected_doc_label_1}段落长度', 'y': f'{selected_doc_label_2}段落长度'}, color=matched_similarities, color_continuous_scale=px.colors.sequential.Viridis)
                                st.plotly_chart(fig_scatter_stats, use_container_width=True)

                            total_segments_A_stats = doc_data_1['segment_count']
                            total_segments_B_stats = doc_data_2['segment_count']
                            matched_count_stats = len(matched_similarities)
                            match_rate_A_stats = (matched_count_stats / total_segments_A_stats * 100) if total_segments_A_stats > 0 else 0
                            match_rate_B_stats = (matched_count_stats / total_segments_B_stats * 100) if total_segments_B_stats > 0 else 0

                            stats_dict_display = {
                                '指标': [f'总段落数 ({selected_doc_label_1})', f'总段落数 ({selected_doc_label_2})', '匹配段落对', f'匹配率 ({selected_doc_label_1})', f'匹配率 ({selected_doc_label_2})', '平均相似度', '最高相似度', '最低相似度'],
                                '数值': [total_segments_A_stats, total_segments_B_stats, matched_count_stats, f"{match_rate_A_stats:.1f}%", f"{match_rate_B_stats:.1f}%", f"{np.mean(matched_similarities):.3f}", f"{np.max(matched_similarities):.3f}", f"{np.min(matched_similarities):.3f}"]
                            }
                            st.dataframe(pd.DataFrame(stats_dict_display), hide_index=True, use_container_width=True)
                        else: st.info("当前选择的文档对之间没有找到匹配的段落。")
                    else: st.info("尚未进行段落对齐分析或无匹配结果。")
    else:
        st.info("👈 请在侧边栏上传并处理文档以开始分析。")


if __name__ == "__main__":
    main()