import streamlit as st
import time
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
DOC_LABELS_AVAILABLE = [chr(65 + i) for i in range(MAX_DOCS)] # 这会自动变成 ['A', 'B']

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
        # 定义更全面的标点符号集合
        # 中文标点符号
        chinese_puncs = '，。、；：？！“”‘’（）【】《》〈〉「」『』〔〕…—～·•◦'
        # 英文标点符号
        english_puncs = ',.!?;:()[]{}\'"`/\\|<>+-_=@#$%^&*'
        # 特殊符号和其他Unicode标点
        special_chars = '©®™§¶†‡°′″‰‱¦¨«»¬¯±²³´µ¶·¸¹º¼½¾¿×÷'
        
        # 创建要移除的字符集合
        chars_to_remove = chinese_puncs + english_puncs + special_chars
        
        # 条件性地添加空格和换行符
        if not keep_space:
            chars_to_remove += ' \t'
        if not keep_newline:
            chars_to_remove += '\n\r'
        
        # 创建并应用转换表 - 比正则表达式更高效
        trans_table = str.maketrans('', '', chars_to_remove)
        return text.translate(trans_table)

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
    def __init__(self):
        # Define standard Chinese opening and closing quotation marks
        self.ALL_POSSIBLE_OPENING_QUOTES = ['“', '‘', '『', '《', '〈']
        self.ALL_POSSIBLE_CLOSING_QUOTES = ['”', '’', '』', '》', '〉']
        
        # Pre-compile regex for matching leading closing quotes for efficiency
        self.leading_closing_quotes_pattern = re.compile(r'^([' + ''.join(re.escape(q) for q in self.ALL_POSSIBLE_CLOSING_QUOTES) + r'])+')
        self.opening_quotes_tuple = tuple(self.ALL_POSSIBLE_OPENING_QUOTES)
        self.closing_quotes_tuple = tuple(self.ALL_POSSIBLE_CLOSING_QUOTES)
        self.extendable_closing_punctuation = ['\u201d', '\u2019', '\u300f', '\u300b', '\u3009', '\uff09', '\u3011', '\u300d', ']', '}']

    def segment_text(self, text: str, min_length: int = 10, delimiters: List[str] = None) -> List[str]:
        if delimiters is None or not delimiters:
            delimiters = ['。', '？', '！', '；', '\n', ' '] # Default, ensure space and newline are here if desired
        
        normalized_text = ChineseTextProcessor.normalize_text(text)
        if not normalized_text.strip():
            return []

        # --- Step 1: Enhanced Preliminary Segmentation with Extended Delimiters ---
        raw_user_delimiters = [d for d in delimiters if d] # Filter out empty strings from user selection
        if not raw_user_delimiters:
            # No valid delimiters, treat as one segment if it meets min_length
            # or if it's the only text available.
            stripped_meaningful_text = ChineseTextProcessor.remove_punctuation(normalized_text).strip()
            if len(stripped_meaningful_text) >= min_length:
                return [normalized_text.strip()]
            elif normalized_text.strip():
                 return [normalized_text.strip()]
            return []

        # Build the list of delimiter patterns to match
        all_delimiter_patterns = []
        # Add extended delimiters (base_delim + closing_punc)
        for base_delim in raw_user_delimiters:
            for closing_punc in self.extendable_closing_punctuation:
                all_delimiter_patterns.append(re.escape(base_delim + closing_punc))
        # Add base delimiters
        for base_delim in raw_user_delimiters:
            all_delimiter_patterns.append(re.escape(base_delim))
        
        # Sort by length descending to prioritize longer matches (e.g., '。\n[CLOSING_QUOTE]' before '。')
        # Remove duplicates first to avoid issues if a base_delim is also an extended_delim part
        unique_delimiter_patterns = sorted(list(set(all_delimiter_patterns)), key=len, reverse=True)
        
        if not unique_delimiter_patterns:
            # This case should ideally not be reached if raw_user_delimiters was not empty,
            # but as a fallback, handle as if no delimiters.
            stripped_meaningful_text = ChineseTextProcessor.remove_punctuation(normalized_text).strip()
            if len(stripped_meaningful_text) >= min_length:
                return [normalized_text.strip()]
            elif normalized_text.strip():
                 return [normalized_text.strip()]
            return []

        split_pattern_str = r'(' + '|'.join(unique_delimiter_patterns) + r')'
        parts = re.split(split_pattern_str, normalized_text)

        preliminary_segments = []
        current_segment_text = ""
        for i, part_content in enumerate(parts):
            if not part_content: # Skip empty parts that can result from re.split
                continue
            
            is_text_part = (i % 2 == 0) # Text parts are at even indices in re.split with capturing group
            
            if is_text_part:
                current_segment_text += part_content
            else: # This is a delimiter part (part_content is the matched delimiter itself)
                current_segment_text += part_content # Append the matched delimiter to the current text
                if current_segment_text.strip(): # Ensure segment is not just whitespace
                    preliminary_segments.append(current_segment_text.strip())
                current_segment_text = "" # Reset for the next segment's text part
        
        # Add any remaining text if the input doesn't end with one of the defined delimiters
        if current_segment_text.strip():
            preliminary_segments.append(current_segment_text.strip())

        if not preliminary_segments and normalized_text.strip(): # Handle case where splitting results in no segments but original text exists
            # This might happen if text contains no delimiters or only delimiters at the very start/end that get stripped.
            # If the text itself (after normalization) is meaningful and meets min_length criteria, treat it as one segment.
            stripped_meaningful_text = ChineseTextProcessor.remove_punctuation(normalized_text).strip()
            if len(stripped_meaningful_text) >= min_length:
                return [normalized_text.strip()]
            elif normalized_text.strip(): # Or if it's shorter but it's all there is
                return [normalized_text.strip()]
            return [] # Otherwise, no valid segments
        elif not preliminary_segments: # Truly no segments and no original text
            return []
        
        # Step 1.5 (handling leading quotes) is now removed as the new splitting logic should cover it.


        # --- Step 2: Merge Short Segments ---
        final_segments = []
        current_merged_segment = ""
        for i, prelim_seg_str in enumerate(preliminary_segments):
            prelim_seg_str = prelim_seg_str.strip()
            if not prelim_seg_str:
                continue

            # Append prelim_seg_str to current_merged_segment
            # Add a space if current_merged_segment is not empty and doesn't end with a delimiter/space,
            # and prelim_seg_str doesn't start with one.
            if current_merged_segment:
                ends_with_sep_or_space = current_merged_segment.endswith(tuple(delimiters) + (" ", "\t"))
                starts_with_sep_or_space = prelim_seg_str.startswith(tuple(delimiters) + (" ", "\t"))
                if not ends_with_sep_or_space and not starts_with_sep_or_space:
                    current_merged_segment += " "
            current_merged_segment += prelim_seg_str

            # Check content length of the current_merged_segment
            content_len = len(ChineseTextProcessor.remove_punctuation(
                current_merged_segment,
                keep_space=(" " not in delimiters), # Only count actual text for length
                keep_newline=("\n" not in delimiters)
            ).strip())

            # Decide if current_merged_segment is complete
            # It's complete if its content length is >= min_length OR it's the last piece of text.
            is_last_prelim = (i == len(preliminary_segments) - 1)

            if content_len >= min_length:
                final_segments.append(current_merged_segment.strip())
                current_merged_segment = ""
            elif is_last_prelim and current_merged_segment.strip(): 
                # If it's the last piece and current_merged_segment has content (but < min_length)
                if final_segments: # Append to the last valid segment
                    # Similar spacing logic for appending
                    ends_with_sep_or_space_final = final_segments[-1].endswith(tuple(delimiters) + (" ", "\t"))
                    starts_with_sep_or_space_current = current_merged_segment.startswith(tuple(delimiters) + (" ", "\t"))
                    if not ends_with_sep_or_space_final and not starts_with_sep_or_space_current:
                       final_segments[-1] += " "
                    final_segments[-1] += current_merged_segment.strip()
                    current_merged_segment = "" # Ensure it's cleared
                else: # This is the only text, even if short
                    final_segments.append(current_merged_segment.strip())
                    current_merged_segment = ""
        
        # Final check for any residue in current_merged_segment (should be empty if logic above is correct)
        if current_merged_segment.strip():
            if final_segments: # Should not happen if last_prelim logic is correct
                final_segments[-1] += (" " + current_merged_segment.strip())
            else:
                final_segments.append(current_merged_segment.strip())

        # Final filter to ensure no empty strings and apply min_length strictly if more than one segment
        output_segments = []
        if len(final_segments) == 1 and final_segments[0]:
            output_segments.append(final_segments[0])
        else:
            for seg in final_segments:
                if seg and len(ChineseTextProcessor.remove_punctuation(
                                seg, 
                                keep_space=(" " not in delimiters), 
                                keep_newline=("\n" not in delimiters)
                                ).strip()) >= min_length:
                    output_segments.append(seg)
                elif output_segments and seg: # If segment is short and there's a previous one, append.
                    # This is a fallback, ideally merging logic should handle this.
                    last_output_seg = output_segments[-1]
                    ends_with_sep_or_space_last_out = last_output_seg.endswith(tuple(delimiters) + (" ", "\t"))
                    starts_with_sep_or_space_seg = seg.startswith(tuple(delimiters) + (" ", "\t"))
                    if not ends_with_sep_or_space_last_out and not starts_with_sep_or_space_seg:
                        output_segments[-1] += " "
                    output_segments[-1] += seg


        return [s for s in output_segments if s.strip()] # Ensure no empty strings from aggressive appends

    def add_segment_ids(self, segments: List[str], doc_id_prefix: str) -> List[Dict]:
        # ... (this method remains the same) ...
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
    # Existing states
    if 'processed_documents_data' not in st.session_state:
        st.session_state.processed_documents_data = {}
    if 'uploaded_file_objects_info' not in st.session_state:
        st.session_state.uploaded_file_objects_info = []
    if 'alignment_table_cache' not in st.session_state:
        st.session_state.alignment_table_cache = {}

    default_delimiter_keys = [
        "句号 (。)", "问号 (？)", "感叹号 (！)", "分号 (；)",
        "换行符 (↵)", "空格 (␣)"
    ]
    if 'selected_delimiters_names' not in st.session_state:
        st.session_state.selected_delimiters_names = default_delimiter_keys

    if 'active_delimiters_chars' not in st.session_state:
        st.session_state.active_delimiters_chars = [
            ALL_POSSIBLE_DELIMITERS[name] for name in st.session_state.selected_delimiters_names
            if name in ALL_POSSIBLE_DELIMITERS
        ]

    # New state variables for refactored workflow
    if 'doc_1_to_compare_label' not in st.session_state:
        st.session_state.doc_1_to_compare_label = None
    if 'doc_2_to_compare_label' not in st.session_state:
        st.session_state.doc_2_to_compare_label = None
    if 'all_docs_processed_at_least_once' not in st.session_state: # Tracks if processing has run for the *selected pair*
        st.session_state.all_docs_processed_at_least_once = False
    
    # Processing flags
    # Note: 'all_docs_processed_at_least_once' is already initialized above (lines 561-562)
    if 'processing_underway' not in st.session_state: 
        st.session_state.processing_underway = False
    if 'process_button_triggered' not in st.session_state:
        st.session_state.process_button_triggered = False
    
    # For parameter change detection - these are updated *before* processing is triggered.
    # Defaults should match the actual defaults of the sidebar widgets.
    if 'last_min_segment_len' not in st.session_state:
        st.session_state.last_min_segment_len = 10 # Default from slider (line ~827)
    if 'last_active_delimiters_chars' not in st.session_state:
        st.session_state.last_active_delimiters_chars = list(st.session_state.active_delimiters_chars)
    if 'last_similarity_thresh' not in st.session_state:
        st.session_state.last_similarity_thresh = 0.3 # Default from slider (line ~848)
    if 'last_chars_per_line_grid' not in st.session_state:
        st.session_state.last_chars_per_line_grid = 30 # Default from slider (line ~851)


ALL_POSSIBLE_DELIMITERS = {
    "句号 (。)": "。", "问号 (？)": "？", "感叹号 (！)": "！", "分号 (；)": "；",
    "逗号 (，)": "，", "顿号 (、)": "、", "冒号 (：)": "：",
    "换行符 (↵)": "\n",  # 使用带箭头的显示名称
    "空格 (␣)": " "     # 使用带空格符号的显示名称
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
            possible_labels_for_dropdown = DOC_LABELS_AVAILABLE[:MAX_DOCS] # Allow choosing from all potential labels A-E

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
                # --- Update dependent session state variables ---
                old_label_of_doc_being_changed = current_assigned_label
                new_label_of_doc_being_changed = new_label_selected

                # 1. Update selected comparison labels
                if st.session_state.doc_1_to_compare_label == old_label_of_doc_being_changed:
                    st.session_state.doc_1_to_compare_label = new_label_of_doc_being_changed
                elif other_doc_index_with_new_label != -1 and st.session_state.doc_1_to_compare_label == new_label_of_doc_being_changed:
                    # The doc selected for comparison 1 was the 'other' doc that got swapped
                    st.session_state.doc_1_to_compare_label = old_label_of_doc_being_changed # It now has this label
                
                if st.session_state.doc_2_to_compare_label == old_label_of_doc_being_changed:
                    st.session_state.doc_2_to_compare_label = new_label_of_doc_being_changed
                elif other_doc_index_with_new_label != -1 and st.session_state.doc_2_to_compare_label == new_label_of_doc_being_changed:
                    # The doc selected for comparison 2 was the 'other' doc that got swapped
                    st.session_state.doc_2_to_compare_label = old_label_of_doc_being_changed # It now has this label

                # 2. Update processed_documents_data keys
                processed_data_temp_store = None
                if old_label_of_doc_being_changed in st.session_state.processed_documents_data:
                    processed_data_temp_store = st.session_state.processed_documents_data.pop(old_label_of_doc_being_changed)
                
                if other_doc_index_with_new_label != -1: # If a swap happened
                    # The 'other' doc (which initially had new_label_of_doc_being_changed) now has old_label_of_doc_being_changed
                    if new_label_of_doc_being_changed in st.session_state.processed_documents_data:
                        data_of_other_doc = st.session_state.processed_documents_data.pop(new_label_of_doc_being_changed)
                        st.session_state.processed_documents_data[old_label_of_doc_being_changed] = data_of_other_doc
                
                if processed_data_temp_store:
                    st.session_state.processed_documents_data[new_label_of_doc_being_changed] = processed_data_temp_store

                # 3. Clear alignment_table_cache (simplest approach for now)
                #    A more sophisticated update would involve iterating and reconstructing keys.
                if st.session_state.alignment_table_cache: # Only clear if not empty
                    st.session_state.alignment_table_cache = {}
                    # Since caches are cleared, reprocessing might be needed if these docs were part of comparison
                    st.session_state.all_docs_processed_at_least_once = False 

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
        
        # --- Sidebar: Select Documents for Comparison ---
        st.sidebar.header("📊 选择对比文档")
        uploaded_doc_labels = [info[1] for info in st.session_state.uploaded_file_objects_info]
        uploaded_doc_display_names_map = {info[1]: info[2] for info in st.session_state.uploaded_file_objects_info}

        if len(uploaded_doc_labels) < 2:
            st.sidebar.info("请至少上传两个文档以进行对比。")
            # Ensure comparison labels are reset if not enough docs
            st.session_state.doc_1_to_compare_label = None
            st.session_state.doc_2_to_compare_label = None
            st.session_state.all_docs_processed_at_least_once = False
        else:
            # Default selection for doc 1
            default_idx1 = 0
            if st.session_state.doc_1_to_compare_label and st.session_state.doc_1_to_compare_label in uploaded_doc_labels:
                default_idx1 = uploaded_doc_labels.index(st.session_state.doc_1_to_compare_label)
            elif uploaded_doc_labels:
                 st.session_state.doc_1_to_compare_label = uploaded_doc_labels[0]
            
            new_doc_1_label = st.sidebar.selectbox(
                "选择文档 1 进行对比:",
                options=uploaded_doc_labels,
                index=default_idx1,
                format_func=lambda x: f"{x} ({uploaded_doc_display_names_map.get(x, '')})",
                key="sidebar_sel_doc_1"
            )
            if new_doc_1_label != st.session_state.doc_1_to_compare_label:
                st.session_state.doc_1_to_compare_label = new_doc_1_label
                st.session_state.all_docs_processed_at_least_once = False # Reset flag on change
                st.rerun() # Rerun to update doc2 options and defaults

            # Default selection for doc 2, ensuring it's different from doc 1
            available_for_doc2 = [l for l in uploaded_doc_labels if l != st.session_state.doc_1_to_compare_label]
            default_idx2 = 0
            if not available_for_doc2:
                 st.session_state.doc_2_to_compare_label = None # No valid option for doc2
            elif st.session_state.doc_2_to_compare_label and st.session_state.doc_2_to_compare_label in available_for_doc2:
                default_idx2 = available_for_doc2.index(st.session_state.doc_2_to_compare_label)
            elif available_for_doc2: # If doc_2_to_compare_label was None or became invalid
                st.session_state.doc_2_to_compare_label = available_for_doc2[0]
                default_idx2 = 0
            
            if available_for_doc2:
                new_doc_2_label = st.sidebar.selectbox(
                    "选择文档 2 进行对比:",
                    options=available_for_doc2,
                    index=default_idx2,
                    format_func=lambda x: f"{x} ({uploaded_doc_display_names_map.get(x, '')})",
                    key="sidebar_sel_doc_2"
                )
                if new_doc_2_label != st.session_state.doc_2_to_compare_label:
                    st.session_state.doc_2_to_compare_label = new_doc_2_label
                    st.session_state.all_docs_processed_at_least_once = False # Reset flag on change
                    st.rerun()
            else:
                st.sidebar.warning("没有其他文档可供选择作为文档2。")
                st.session_state.doc_2_to_compare_label = None

        can_process = (st.session_state.doc_1_to_compare_label is not None and 
                       st.session_state.doc_2_to_compare_label is not None and 
                       st.session_state.doc_1_to_compare_label != st.session_state.doc_2_to_compare_label and 
                       len(st.session_state.uploaded_file_objects_info) >= 2)

        process_all_button = st.sidebar.button("🚀 处理所有已上传文档", type="primary", use_container_width=True,
                                             disabled=not can_process)

    # --- Sidebar: Processing Parameters ---
    st.sidebar.header("⚙️ 处理参数")
    st.sidebar.markdown("<h4 style='font-size: 1em; margin-bottom: 0.1em;'>最小段落长度 (有效字符)</h4>", unsafe_allow_html=True)
    min_segment_len = st.sidebar.slider("最小段落长度 (有效字符)", 3, 50, 10, 1, help="段落分割后，每个段落最少包含的有效字符数（不计标点）。", label_visibility="collapsed")
    
    st.sidebar.markdown("<h4 style='font-size: 1em; margin-bottom: 0.1em;'>段落分割符选择</h4>", unsafe_allow_html=True)
    temp_selected_delimiters_names = []
    for name, char_val in ALL_POSSIBLE_DELIMITERS.items():
        col1, col2 = st.sidebar.columns([1, 8]) # Adjust column ratios as needed
        with col1:
            # Use a unique key for the checkbox, label_visibility collapsed hides the actual label string
            is_checked = st.checkbox("", value=(name in st.session_state.selected_delimiters_names), key=f"delim_cb_{name}", label_visibility="collapsed")
        with col2:
            # Display the styled name using markdown. Adjust vertical-align if needed.
            st.markdown(f"<span style='font-size: 0.85em; display: inline-block; margin-top: 5px;'>{name}</span>", unsafe_allow_html=True)
        
        if is_checked:
            temp_selected_delimiters_names.append(name)
    
    if temp_selected_delimiters_names != st.session_state.selected_delimiters_names:
        st.session_state.selected_delimiters_names = temp_selected_delimiters_names
        st.session_state.active_delimiters_chars = [ALL_POSSIBLE_DELIMITERS[name] for name in st.session_state.selected_delimiters_names if name in ALL_POSSIBLE_DELIMITERS]

    st.sidebar.markdown("<h4 style='font-size: 1em; margin-bottom: 0.1em;'>相似度阈值 (段落匹配)</h4>", unsafe_allow_html=True)
    similarity_thresh = st.sidebar.slider("相似度阈值", 0.1, 0.99, 0.3, 0.01, label_visibility="collapsed")
    
    st.sidebar.markdown("<h4 style='font-size: 1em; margin-bottom: 0.1em;'>网格每行字符数 (字符网格)</h4>", unsafe_allow_html=True)
    chars_per_line_grid = st.sidebar.slider("网格每行字符数", 10, 80, 30, 1, label_visibility="collapsed")

    # --- Check for Explicit Button Trigger for Processing ---
    if st.session_state.get('process_button_triggered', False):
        st.session_state.process_button_triggered = False # Consume the flag
        # Parameters should have been set to last_... when the button was clicked before rerun
        st.session_state.processing_underway = True
        st.session_state.processed_documents_data = {}
        st.session_state.alignment_table_cache = {}
        st.session_state.all_docs_processed_at_least_once = False
        # No rerun here, allow script to flow into processing block

    # --- Main Processing Logic ---
    # Initialize session state variables if they don't exist
    if 'processing_underway' not in st.session_state: # To show spinner status
        st.session_state.processing_underway = False

    # Parameter change detection
    params_changed = (
        min_segment_len != st.session_state.get('last_min_segment_len') or
        set(st.session_state.active_delimiters_chars) != set(st.session_state.get('last_active_delimiters_chars', [])) or
        similarity_thresh != st.session_state.get('last_similarity_thresh') or
        chars_per_line_grid != st.session_state.get('last_chars_per_line_grid')
    )

    if params_changed and not st.session_state.processing_underway:
        # Only trigger processing due to param change if not already triggered by button for these params
        st.session_state.all_docs_processed_at_least_once = False # Indicate re-processing is needed for current view
        if st.session_state.uploaded_file_objects_info and \
           st.session_state.doc_1_to_compare_label and st.session_state.doc_2_to_compare_label and \
           st.session_state.doc_1_to_compare_label != st.session_state.doc_2_to_compare_label:
            # If a valid pair is selected, and params changed, set to process
            st.session_state.processing_underway = True 
            st.session_state.processed_documents_data = {} # Clear old processed data
            st.session_state.alignment_table_cache = {}   # Clear alignment cache
            # Update last_... params HERE to prevent rerun loop and reflect current settings for processing
            st.session_state.last_min_segment_len = min_segment_len
            st.session_state.last_active_delimiters_chars = list(st.session_state.active_delimiters_chars) # Ensure a copy
            st.session_state.last_similarity_thresh = similarity_thresh
            st.session_state.last_chars_per_line_grid = chars_per_line_grid
            st.rerun() # Rerun to trigger processing

    # If "Process All" button is clicked by the user
    if 'process_all_button' in locals() and process_all_button: # process_all_button is defined in sidebar
        if st.session_state.doc_1_to_compare_label and st.session_state.doc_2_to_compare_label and \
           st.session_state.doc_1_to_compare_label != st.session_state.doc_2_to_compare_label:
            st.session_state.process_button_triggered = True # Set the trigger flag
            # Update last_... params HERE to capture the settings at the moment of button click
            st.session_state.last_min_segment_len = min_segment_len
            st.session_state.last_active_delimiters_chars = list(st.session_state.active_delimiters_chars) # Ensure a copy
            st.session_state.last_similarity_thresh = similarity_thresh
            st.session_state.last_chars_per_line_grid = chars_per_line_grid
            # No need to clear caches or set processing_underway here; the new block will handle it after rerun
            st.rerun() # Rerun to activate the trigger
        else:
            st.sidebar.error("请先在侧边栏选择两个不同的文档进行对比。")

    # --- Main Content Display --- 
    # This entire block replaces the old main content display logic from original line 800 to 1060

    # 1. Handle initial states: no uploads, or docs not selected for comparison
    if not st.session_state.uploaded_file_objects_info:
        st.info("👈 请在侧边栏上传文档以开始。")
    elif not st.session_state.doc_1_to_compare_label or not st.session_state.doc_2_to_compare_label:
        st.info("👈 请在侧边栏选择两个文档进行对比分析。")
    elif st.session_state.doc_1_to_compare_label == st.session_state.doc_2_to_compare_label:
        st.warning("👈 请在侧边栏选择两个不同的文档进行对比。")
    
    # 2. Handle processing state or if processing is needed
    elif st.session_state.processing_underway:
        # This spinner will show if processing_underway was set true before a rerun
        with st.spinner("正在处理文档，请稍候..."):
            # The actual processing logic is now placed here to ensure it runs within the spinner context
            # and directly updates state before the next display check.
            # No need to check for processing trigger here, as processing_underway is already True
            # Use current parameter values from the sidebar for processing
            current_min_segment_len = min_segment_len 
            current_active_delimiters_chars = st.session_state.active_delimiters_chars
            current_similarity_thresh = similarity_thresh 
            current_chars_per_line_grid = chars_per_line_grid

            for file_obj, assigned_label, original_name in st.session_state.uploaded_file_objects_info:
                try:
                    file_obj.seek(0) # Reset read pointer for re-processing
                    raw_text = ChineseTextProcessor.extract_text_from_file(file_obj)
                    normalized_text = ChineseTextProcessor.normalize_text(raw_text)
                    segmenter = ChineseSegmenter()
                    segments_text = segmenter.segment_text(normalized_text, min_length=current_min_segment_len, delimiters=current_active_delimiters_chars)
                    segments_data_with_ids = segmenter.add_segment_ids(segments_text, doc_id_prefix=assigned_label)
                    
                    st.session_state.processed_documents_data[assigned_label] = {
                        'doc_label': assigned_label,
                        'filename': original_name,
                        'full_raw_text': raw_text,
                        'full_normalized_text': normalized_text,
                        'segments_text': segments_text,
                        'segments_data': segments_data_with_ids,
                        'segment_count': len(segments_text)
                    }
                except Exception as e:
                    st.error(f"处理文档 {original_name} (Label {assigned_label}) 时出错: {e}")
            
            # Processing finished successfully
            st.session_state.processing_underway = False 
            st.session_state.all_docs_processed_at_least_once = True # Mark that processing for this selection is done
            
            # Update last parameters *after* successful processing
            st.session_state.last_min_segment_len = current_min_segment_len
            st.session_state.last_active_delimiters_chars = list(current_active_delimiters_chars)
            st.session_state.last_similarity_thresh = current_similarity_thresh
            st.session_state.last_chars_per_line_grid = current_chars_per_line_grid
            
            st.rerun() # Rerun to update display with processed data

    # 3. Display comparison results if documents are selected and processed
    elif st.session_state.doc_1_to_compare_label and st.session_state.doc_2_to_compare_label and \
         st.session_state.all_docs_processed_at_least_once:
        
        selected_doc_label_1 = st.session_state.doc_1_to_compare_label
        selected_doc_label_2 = st.session_state.doc_2_to_compare_label

        doc_data_1 = st.session_state.processed_documents_data.get(selected_doc_label_1)
        doc_data_2 = st.session_state.processed_documents_data.get(selected_doc_label_2)

        if doc_data_1 and doc_data_2:
            # --- Tabs for Alignment Table, Grid, Stats (Pairwise) ---
            tab_align, tab_grid, tab_stats = st.tabs(["📜 对齐表格视图", "⠿ 字符网格对比", "📈 统计图表"])

            with tab_align:
                st.markdown(f"#### 段落对齐: **{doc_data_1['filename']} ({selected_doc_label_1})** vs **{doc_data_2['filename']} ({selected_doc_label_2})**")
                cache_key_align = tuple(sorted((selected_doc_label_1, selected_doc_label_2))) + \
                                  (similarity_thresh, min_segment_len, tuple(sorted(st.session_state.active_delimiters_chars)))
                
                if cache_key_align not in st.session_state.alignment_table_cache:
                    with st.spinner(f"计算 {selected_doc_label_1} 和 {selected_doc_label_2} 的段落对齐..."):
                        matcher_align = ChineseSimilarityMatcher()
                        st.session_state.alignment_table_cache[cache_key_align] = matcher_align.find_best_matches(
                            doc_data_1['segments_data'], doc_data_2['segments_data'], similarity_thresh # Use current similarity_thresh from sidebar
                        )
                
                current_alignment_table = st.session_state.alignment_table_cache[cache_key_align]
                matched_count_pair = sum(1 for item in current_alignment_table if item.get('matched', False))
                st.info(f"找到 {matched_count_pair} 对相似段落 (阈值 > {similarity_thresh:.2f})。内容预览限100字符。")

                df_data_align = []
                sorted_alignment_for_df = sorted(current_alignment_table, key=lambda x: (x['a_index'] if x['a_index'] != -1 else float('inf'), x['b_index'] if x['b_index'] != -1 else float('inf')))

                for item in sorted_alignment_for_df:
                    row = {}
                    if item['segment_a']:
                        row[f'{selected_doc_label_1}段落ID'] = item['segment_a']['id']
                        row[f'{selected_doc_label_1}段落内容'] = item['segment_a']['text'][:100] + ('...' if len(item['segment_a']['text']) > 100 else '')
                    else:
                        row[f'{selected_doc_label_1}段落ID'] = ''
                        row[f'{selected_doc_label_1}段落内容'] = ''
                    
                    if item['segment_b']:
                        row[f'{selected_doc_label_2}段落ID'] = item['segment_b']['id']
                        row[f'{selected_doc_label_2}段落内容'] = item['segment_b']['text'][:100] + ('...' if len(item['segment_b']['text']) > 100 else '')
                    else:
                        row[f'{selected_doc_label_2}段落ID'] = ''
                        row[f'{selected_doc_label_2}段落内容'] = ''
                    row['相似度'] = f"{item['similarity']:.3f}" if item['matched'] else "-"
                    df_data_align.append(row)
                
                df_display_align = pd.DataFrame(df_data_align)
                if not df_display_align.empty and '相似度' in df_display_align.columns:
                    cols_order = ['相似度'] + [col for col in df_display_align.columns if col != '相似度']
                    df_display_align = df_display_align[cols_order]
                
                st.dataframe(df_display_align, use_container_width=True, height=500)
                
                # Use more unique keys for download buttons to avoid conflicts after reruns
                csv_export_align = df_display_align.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(f"📥 下载 {selected_doc_label_1}-{selected_doc_label_2} 对齐表 (CSV)", csv_export_align, f"alignment_{selected_doc_label_1}_{selected_doc_label_2}.csv", "text/csv", key=f"csv_dl_{selected_doc_label_1}_{selected_doc_label_2}_{int(time.time())}")
                
                if not df_display_align.empty:
                    output_xlsx = io.BytesIO()
                    with pd.ExcelWriter(output_xlsx, engine='openpyxl') as writer:
                        df_display_align.to_excel(writer, index=False, sheet_name='Alignment')
                    excel_data = output_xlsx.getvalue()
                    st.download_button(f"📥 下载 {selected_doc_label_1}-{selected_doc_label_2} 对齐表 (XLSX)", data=excel_data, file_name=f"alignment_{selected_doc_label_1}_{selected_doc_label_2}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"xlsx_dl_{selected_doc_label_1}_{selected_doc_label_2}_{int(time.time())}")

            with tab_grid:
                st.markdown(f"#### 字符网格: **{doc_data_1['filename']} ({selected_doc_label_1})** vs **{doc_data_2['filename']} ({selected_doc_label_2})**")
                ignore_punc_grid_display = st.checkbox("比对和显示时忽略所有标点符号", value=False, key="cb_ignore_punc_grid")
                text_A_for_grid = doc_data_1['full_raw_text']
                text_B_for_grid = doc_data_2['full_raw_text']
                with st.spinner("生成字符网格..."):
                    aligned_A_grid_chars, aligned_B_grid_chars = CharacterGridAligner.align_full_texts_for_grid(text_A_for_grid, text_B_for_grid, ignore_punc_grid_display)
                
                if aligned_A_grid_chars or aligned_B_grid_chars:
                    grid_html_content = CharacterGridAligner.create_grid_html_for_full_docs(selected_doc_label_1, selected_doc_label_2, aligned_A_grid_chars, aligned_B_grid_chars, chars_per_line_grid) # Use current chars_per_line_grid from sidebar
                    st.components.v1.html(grid_html_content, height=700, scrolling=True)
                    st.download_button(f"📥 下载 {selected_doc_label_1}-{selected_doc_label_2} 网格 (HTML)", grid_html_content, f"grid_{selected_doc_label_1}_{selected_doc_label_2}.html", "text/html", key=f"html_dl_{selected_doc_label_1}_{selected_doc_label_2}_{int(time.time())}")
                else:
                    st.info("选择的文档内容为空或处理后为空，无法生成字符网格。")

            with tab_stats:
                st.markdown(f"#### 文档对齐分布图: **{doc_data_1['filename']} ({selected_doc_label_1})** vs **{doc_data_2['filename']} ({selected_doc_label_2})**")
                current_alignment_table_for_dist = st.session_state.alignment_table_cache.get(cache_key_align, []) # Uses same cache key as tab_align
                if not current_alignment_table_for_dist:
                    st.info("请先在“对齐表格视图”标签页生成对齐数据，或当前参数下无对齐结果。")
                else:
                    plot_data = []
                    max_index_a = 0
                    max_index_b = 0
                    for item in current_alignment_table_for_dist:
                        if item.get('matched', False) and item['segment_a'] and item['segment_b']:
                            doc_a_idx_1based = item['segment_a']['index'] + 1
                            doc_b_idx_1based = item['segment_b']['index'] + 1
                            plot_data.append({'doc_A_index': doc_a_idx_1based, 'doc_B_index': doc_b_idx_1based, 'similarity': item['similarity'], 'type': 'Matched Pair', 'text_A': item['segment_a']['text'][:30] + "...", 'text_B': item['segment_b']['text'][:30] + "...", 'id_A': item['segment_a']['id'], 'id_B': item['segment_b']['id']})
                            max_index_a = max(max_index_a, doc_a_idx_1based)
                            max_index_b = max(max_index_b, doc_b_idx_1based)
                        elif item['segment_a'] and not item['segment_b']:
                            doc_a_idx_1based = item['segment_a']['index'] + 1
                            plot_data.append({'doc_A_index': doc_a_idx_1based, 'doc_B_index': 0, 'similarity': 0, 'type': f'{selected_doc_label_1} Only', 'text_A': item['segment_a']['text'][:30] + "...", 'text_B': "", 'id_A': item['segment_a']['id'], 'id_B': ""})
                            max_index_a = max(max_index_a, doc_a_idx_1based)
                        elif item['segment_b'] and not item['segment_a']:
                            doc_b_idx_1based = item['segment_b']['index'] + 1
                            plot_data.append({'doc_A_index': 0, 'doc_B_index': doc_b_idx_1based, 'similarity': 0, 'type': f'{selected_doc_label_2} Only', 'text_A': "", 'text_B': item['segment_b']['text'][:30] + "...", 'id_A': "", 'id_B': item['segment_b']['id']})
                            max_index_b = max(max_index_b, doc_b_idx_1based)
                    
                    if not plot_data:
                        st.info("没有可供可视化的对齐数据。")
                    else:
                        df_plot = pd.DataFrame(plot_data)
                        available_colorscales = ['Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis', 'Pinkyl', 'Blues', 'Greens', 'Reds', 'YlOrRd', 'YlGnBu', 'Hot', 'Electric', 'Rainbow', 'Plotly3']
                        if 'selected_colorscale' not in st.session_state:
                            st.session_state.selected_colorscale = "Pinkyl"
                        st.session_state.selected_colorscale = st.selectbox("选择配色方案:", available_colorscales, index=available_colorscales.index(st.session_state.selected_colorscale), key="scatter_colorscale_selector")
                        
                        fig_dist = px.scatter(df_plot, x='doc_A_index', y='doc_B_index', color='similarity', 
                                            hover_data=['id_A', 'id_B', 'text_A', 'text_B', 'similarity'], 
                                            color_continuous_scale=st.session_state.selected_colorscale, range_color=[0,1],
                                            title="段落对齐分布图", 
                                            labels={'doc_A_index': f'文档 {selected_doc_label_1} 段落序号', 
                                                    'doc_B_index': f'文档 {selected_doc_label_2} 段落序号', 
                                                    'similarity': '相似度'})
                        fig_dist.update_traces(marker=dict(size=5))
                        current_max_a = max(1, max_index_a); current_max_b = max(1, max_index_b)
                        fig_dist.update_xaxes(range=[0, current_max_a + 1]); fig_dist.update_yaxes(range=[0, current_max_b + 1])
                        if max_index_a > 0 or max_index_b > 0:
                            max_val_for_diag = max(max_index_a, max_index_b, 0)
                            fig_dist.add_shape(type="line", x0=1, y0=1, x1=max_val_for_diag, y1=max_val_for_diag, line=dict(color="rgba(128, 128, 128, 0.5)", width=1, dash="dash"))
                        fig_dist.update_layout(height=600, legend_title_text='段落类型', margin=dict(l=40, r=40, t=50, b=40))
                        st.plotly_chart(fig_dist, use_container_width=True)
                        st.markdown("""**图表解读:**
                        - **散点位置:** 每个点代表一个段落（或一对匹配的段落）。X轴和Y轴分别表示其在文档A和文档B中的序号。
                        - **颜色深浅:** 颜色越深（或根据色阶变化），表示匹配段落间的相似度越高。
                        - **对角线参考:** 图中可能包含一条虚线对角线 (y=x)。靠近这条线的点表明段落在两个文档中的相对顺序相似。
                        - **悬停信息:** 将鼠标悬停在点上可以查看段落ID、内容片段和具体相似度得分。
                        - **仅在一个文档中出现的段落:** 可能会显示在X轴或Y轴的零点附近（或图表边缘），表示它们在另一文档中没有对应匹配。""")
        elif not st.session_state.all_docs_processed_at_least_once : # Data not found, and processing hasn't happened for this selection
             st.info("👈 请点击侧边栏的“处理所有已上传文档”按钮以使用当前设置生成对比结果。")
        else: # Data not found, but all_docs_processed_at_least_once is true (e.g. error during processing, or data cleared unexpectedly)
             st.warning("选定文档的数据不存在或不完整。请尝试重新点击“处理所有已上传文档”。")
    
    # Fallback for any other unhandled state
    else:
        if not st.session_state.processing_underway: # Avoid showing this if a spinner is already active
            st.info("👈 请在侧边栏进行操作以开始分析。")


if __name__ == "__main__":
    main()