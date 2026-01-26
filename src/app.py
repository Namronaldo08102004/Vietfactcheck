import sys
import os
import streamlit as st
import json
import random
import torch

# --- XỬ LÝ PATH HỆ THỐNG ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.settings import settings
from src.components.vectorDB import VietnameseVectorDB
from src.components.reranker import VietnameseReranker
from src.modules.document_retrieval import DocumentRetrievalModule
from src.modules.evidence_selection import EvidenceSelectionModule
from src.modules.claim_verification import ClaimVerificationModule
from src.modules.claim_extraction import BERTSumClaimExtractor

import src.components.presumm.model as _models
sys.modules["models"] = _models

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="VietFactCheck System", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .stButton>button { 
        width: 100%; 
        border-radius: 8px; 
        border: 1px solid #ff4b4b; 
        font-weight: bold;
        transition: 0.3s;
        margin-bottom: 10px;
    }
    .stButton>button:hover {
        background-color: #ff4b4b;
        color: white;
    }
    .highlight { 
        background-color: #fff2cc; 
        border: 1px solid #ffd966; 
        padding: 2px; 
        border-radius: 4px; 
        color: #333;
        font-weight: 500;
    }
    .claim-step {
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        border-left: 5px solid #ff4b4b;
        background-color: #f9f9f9;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ Hệ thống Xác thực Thông tin Tiếng Việt")

TOPIC_ICONS = {
    'khoa học': '🧪', 'văn hoá': '🎨', 'văn hóa': '🎨', 'quân sự': '🛡️', 'khoa giáo': '📚',
    'kinh doanh': '💼', 'chính trị': '🏛️', 'thế giới': '🌍', 'thời sự': '🗞️', 'sức khoẻ': '🏥',
    'sức khỏe': '🏥', 'đời sống': '🌱', 'giải trí': '🎬', 'hoa hậu': '👑', 'kinh tế': '📈',
    'an ninh trật tự': '👮', 'pháp luật': '⚖️', 'thể thao': '⚽', 'du lịch': '✈️', 'địa phương': '📍',
    'giới trẻ': '🌈', 'bất động sản': '🏠', 'giáo dục': '🎓', 'số hóa': '🔢', 'người lính': '🎖️',
    'nhịp sống phương nam': '🏙️', 'xã hội': '👥', 'quốc tế': '🌐', 'y tế': '💉', 'địa ốc': '🏗️',
    'đô thị': '🌆', 'công nghệ': '💻', 'khoa học công nghệ': '🚀', 'nhà đất': '🏡', 
    'giáo dục - hướng nghiệp': '📖', 'bạn đọc làm báo': '✍️', 'văn hóa - xã hội': '🎭'
}

# --- HÀM KHỞI TẠO HỆ THỐNG ---
@st.cache_data
def load_recommendations():
    path = settings.DATA_PATHS.get("train")
    recs = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            random.shuffle(data)
            for item in data:
                topic = item.get("Topic", "khác").strip().lower()
                if topic not in recs: recs[topic] = item.get("Statement", "")
    return recs

@st.cache_data
def load_news_recommendations():
    path = settings.EXTRACTION_DATA_PATHS.get("train")
    recs = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            random.shuffle(data)
            for item in data:
                topic = item.get("topic", "khác").strip().lower()
                if topic not in recs: recs[topic] = item.get("fake_context", "")
    return recs

@st.cache_resource
def init_core_system():
    db = VietnameseVectorDB("master_db", settings.STORAGE_DIR, 
                            settings.EMBEDDING_MODEL, settings.TRUNCATION_DIM)
    ret_mod = DocumentRetrievalModule(db)
    if not db.load():
        ret_mod.build_system(list(settings.DATA_PATHS.values()))
    
    url_map = {}
    for p in settings.DATA_PATHS.values():
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                for item in json.load(f): url_map[item['Url']] = item['Context']
    
    extractor = BERTSumClaimExtractor(
        model_path = getattr(settings, "EXTRACTOR_MODEL_PATH", "bertext_cnndm_transformer.pt"),
        visible_gpus = "-1" if torch.cuda.is_available() == False else "0"
    )
                
    return ret_mod, EvidenceSelectionModule(db), url_map, VietnameseReranker(), extractor

ret_mod, ev_mod, url_to_context, reranker, extractor = init_core_system()
recs_dict = load_recommendations()
news_dict = load_news_recommendations()

if "main_input" not in st.session_state: st.session_state["main_input"] = ""
if "rec_mode" not in st.session_state: st.session_state["rec_mode"] = "claim" 

# --- SIDEBAR: ĐIỀU KHIỂN ---
st.sidebar.title("🎮 Control Panel")
target_stage = st.sidebar.selectbox("Giai đoạn dừng xử lý:", 
                                    ["Document Retrieval", "Evidence Selection", "Claim Verification"])

st.sidebar.subheader("🎨 Giao diện gợi ý")
grid_cols = st.sidebar.slider("Số cột hiển thị Topic:", 2, 8, 6)

# 1. Document Retrieval Settings
with st.sidebar.expander("1. Document Retrieval Settings", expanded=True):
    dr_w_emb = st.slider("Embedding Weight", 0.0, 1.0, 0.4, key="dr_emb")
    dr_w_bm25 = st.slider("BM25 Weight", 0.0, 1.0, 0.3, key="dr_bm25")
    dr_w_tfidf = 1.0 - dr_w_emb - dr_w_bm25
    st.slider("TF-IDF Weight (Cố định)", 0.0, 1.0, max(0.0, dr_w_tfidf), disabled=True)
    
    # --- YÊU CẦU 2: Cảnh báo UI cho DR ---
    if dr_w_emb + dr_w_bm25 > 1.0:
        st.error("⚠️ Tổng trọng số vượt quá 1.0!")

    dr_use_rerank = st.toggle("Sử dụng Reranker cho Document?")
    dr_top_k = st.number_input("Top K URLs", 1, 10, 3 if dr_use_rerank else 1)

MODEL_MAPPING = {
    "XLM-RoBERTa-base": "Vifactcheck-xlm-roberta-base",
    "XLM-RoBERTa-large": "Vifactcheck-xlm-roberta-large",
    "ViBERT": "Vifactcheck-ViBERT",
    "mBERT": "Vifactcheck-mBERT",
    "PhoBERT-base": "Vifactcheck-phoBERT-base",
    "PhoBERT-large": "Vifactcheck-phoBERT-large"
}

v_mode = "Selected Evidences"
selected_hf_model = ""

if target_stage == "Claim Verification":
    with st.sidebar.expander("3. Claim Verification Settings", expanded=True):
        v_mode = st.radio("Xác thực dựa trên:", ["Full Context", "Selected Evidences"])
        display_model_name = st.selectbox("Chọn Model PLM:", list(MODEL_MAPPING.keys()))
        base_name = MODEL_MAPPING[display_model_name]
        suffix = "-gold-evidence" if v_mode == "Selected Evidences" else ""
        selected_hf_model = f"Namronaldo2004/{base_name}{suffix}"

show_ev = (target_stage == "Evidence Selection") or (target_stage == "Claim Verification" and v_mode == "Selected Evidences")
if show_ev:
    with st.sidebar.expander("2. Evidence Selection Settings", expanded=True):
        ev_w_emb = st.slider("Evid. Embedding Weight", 0.0, 1.0, 0.6, key="ev_emb")
        ev_w_bm25 = st.slider("Evid. BM25 Weight", 0.0, 1.0, 0.2, key="ev_bm25")
        
        # --- YÊU CẦU 1: Bổ sung hiển thị TF-IDF Weight cho Evidence ---
        ev_w_tfidf = 1.0 - ev_w_emb - ev_w_bm25
        st.slider("Evid. TF-IDF Weight (Cố định)", 0.0, 1.0, max(0.0, ev_w_tfidf), disabled=True)
        
        # --- YÊU CẦU 2: Cảnh báo UI cho Evidence ---
        if ev_w_emb + ev_w_bm25 > 1.0:
            st.error("⚠️ Tổng trọng số vượt quá 1.0!")

        ev_use_rerank = st.toggle("Sử dụng Reranker cho Evidence?", value=True)
        if ev_use_rerank:
            ev_top_k_input = st.number_input("Số lượng bằng chứng trước Rerank:", 3, 20, 10)
            t1 = st.slider("Confidence Threshold (T1)", 0.6, 1.0, 0.75)
            t2 = st.slider("Gap Threshold (T2)", 0.0, 0.15, 0.05)
        else:
            ev_top_k_input = st.number_input("Số lượng bằng chứng (Top K):", 1, 10, 3)

# --- KHU VỰC GỢI Ý ---
col_title, col_nav = st.columns([0.8, 0.2])
with col_title:
    if st.session_state["rec_mode"] == "claim":
        st.subheader("💡 Gợi ý Claim theo chủ đề")
        current_data = recs_dict
    else:
        st.subheader("📰 Gợi ý bản tin thời sự theo chủ đề")
        current_data = news_dict

with col_nav:
    if st.session_state["rec_mode"] == "claim":
        if st.button("Tiếp theo ➡️"):
            st.session_state["rec_mode"] = "news"
            st.rerun()
    else:
        if st.button("⬅️ Quay lại"):
            st.session_state["rec_mode"] = "claim"
            st.rerun()

topic_list = list(current_data.keys())
for i in range(0, len(topic_list), grid_cols):
    cols = st.columns(grid_cols)
    for j in range(grid_cols):
        if i + j < len(topic_list):
            topic = topic_list[i + j]
            icon = TOPIC_ICONS.get(topic, '📝')
            if cols[j].button(f"{icon} {topic.capitalize()}", key=f"btn_{topic}_{st.session_state['rec_mode']}"):
                st.session_state["main_input"] = current_data[topic]
                st.rerun()

st.divider()

# --- GIAO DIỆN CHÍNH ---
claim_text = st.text_area("Nhập nội dung cần kiểm chứng (Claim):", key="main_input", height=150)
use_extraction = st.checkbox("Chia nhỏ nội dung đầu vào thành các claim riêng biệt để kiểm chứng", value=False)

if st.button("🚀 Bắt đầu thực hiện xử lý", type="primary"):
    # --- YÊU CẦU 2: Kiểm tra trọng số trước khi xử lý ---
    if dr_w_emb + dr_w_bm25 > 1.0:
        st.error("❌ Không thể thực hiện: Tổng trọng số Document Retrieval vượt quá 1.0. Vui lòng điều chỉnh lại ở thanh bên!")
        st.stop()
    
    if show_ev and (ev_w_emb + ev_w_bm25 > 1.0):
        st.error("❌ Không thể thực hiện: Tổng trọng số Evidence Selection vượt quá 1.0. Vui lòng điều chỉnh lại ở thanh bên!")
        st.stop()

    if not claim_text.strip():
        st.warning("Vui lòng nhập nội dung!")
        st.stop()

    # Xử lý danh sách Claim
    claims_to_process = []
    if use_extraction:
        with st.spinner("✂️ Đang phân tách nội dung..."):
            claims_to_process = extractor.extract(claim_text)
            if not claims_to_process:
                st.error("Không thể tách được claim nào. Sử dụng nội dung gốc.")
                claims_to_process = [claim_text]
            else:
                st.info(f"✅ Đã tìm thấy **{len(claims_to_process)}** claim cần xác thực.")
    else:
        claims_to_process = [claim_text]

    claim_tabs = st.tabs([f"Claim {i+1}" for i in range(len(claims_to_process))])

    for idx, (current_claim, tab) in enumerate(zip(claims_to_process, claim_tabs)):
        with tab:
            st.markdown(f"**Nội dung kiểm chứng:** *{current_claim}*")
            
            # --- BƯỚC 1: DOCUMENT RETRIEVAL ---
            with st.status(f"🔍 [C{idx+1}] Đang truy xuất bài báo...") as s:
                dr_weights = (dr_w_bm25, 1.0 - dr_w_emb - dr_w_bm25, dr_w_emb)
                urls = ret_mod.get_top_k_url(current_claim, top_k=dr_top_k, weights=dr_weights)
                
                if dr_use_rerank:
                    class Item:
                        def __init__(self, url, content): 
                            self.url, self.page_content = url, content
                    cands = [Item(u, url_to_context[u]) for u in urls]
                    best_url = reranker.rerank(current_claim, cands)[0]['document'].url
                else:
                    best_url = urls[0]
                s.update(label="✅ Đã tìm thấy nguồn!", state="complete")

            st.markdown(f"**Nguồn:** [{best_url}]({best_url})")
            full_text = url_to_context.get(best_url, "")

            if target_stage == "Document Retrieval":
                st.write(full_text)
                continue

            # --- BƯỚC 2: EVIDENCE SELECTION ---
            selected_evidences = []
            if show_ev:
                with st.status(f"📍 [C{idx+1}] Đang trích xuất bằng chứng...") as s:
                    ev_weights = (ev_w_bm25, 1.0 - ev_w_emb - ev_w_bm25, ev_w_emb)
                    cands = ev_mod.select_top_k_evidence(current_claim, best_url, top_k=ev_top_k_input, weights=ev_weights)
                    
                    if ev_use_rerank:
                        reranked_ev = reranker.rerank(current_claim, cands)
                        high_score = [res for res in reranked_ev if res['rerank_score'] > t1]
                        if high_score:
                            selected_evidences = [res['document'] for res in high_score]
                        else:
                            selected_evidences = [reranked_ev[0]['document']]
                            for i in range(1, len(reranked_ev)):
                                if (reranked_ev[i-1]['rerank_score'] - reranked_ev[i]['rerank_score']) < t2:
                                    selected_evidences.append(reranked_ev[i]['document'])
                                else: break
                    else:
                        selected_evidences = cands[:ev_top_k_input]
                    s.update(label=f"✅ {len(selected_evidences)} bằng chứng!", state="complete")

                highlighted_html = full_text
                for ev in selected_evidences:
                    snippet = ev.page_content.strip()
                    highlighted_html = highlighted_html.replace(snippet, f'<span class="highlight">{snippet}</span>')
                st.markdown(f"<div style='text-align: justify;'>{highlighted_html}</div>", unsafe_allow_html=True)
            else:
                st.write(full_text)

            if target_stage == "Evidence Selection":
                continue

            # --- BƯỚC 3: CLAIM VERIFICATION ---
            if target_stage == "Claim Verification":
                with st.spinner(f"⚖️ Đang xác thực Claim {idx+1}..."):
                    verifier = ClaimVerificationModule(selected_hf_model)
                    result = verifier.verify_claim(
                        current_claim, 
                        full_context=full_text if v_mode == "Full Context" else None,
                        evidences=selected_evidences if v_mode == "Selected Evidences" else None
                    )
                    
                    st.divider()
                    label = result['label_name']
                    if label == "Supported":
                        st.success(f"✅ **CHÍNH XÁC**")
                    elif label == "Refuted":
                        st.error(f"❌ **SAI SỰ THẬT**")
                    else:
                        st.warning(f"❓ **KHÔNG ĐỦ THÔNG TIN**")