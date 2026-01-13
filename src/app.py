import sys
import os
import streamlit as st
import json
import random

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

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="VietFactCheck System", layout="wide", initial_sidebar_state="expanded")

# CSS tùy chỉnh để làm đẹp các ô gợi ý và highlight bằng chứng
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
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ Hệ thống Xác thực Thông tin Tiếng Việt")

# Icon cho 36 topic từ dataset ViFactCheck
TOPIC_ICONS = {
    'khoa học': '🧪', 'văn hoá': '🎨', 'văn hóa': '🎨', 'quân sự': '🛡️', 'khoa giáo': '📚',
    'kinh doanh': '💼', 'chính trị': '🏛️', 'thế giới': '🌍', 'thời sự': '🗞️', 'sức khoẻ': '🏥',
    'sức khỏe': '🏥', 'đời sống': '🌱', 'giải trí': '🎬', 'hoa hậu': '👑', 'kinh tế': '📈',
    'an ninh trật tự': '👮', 'pháp luật': '⚖️', 'thể thao': '⚽', 'du lịch': '✈️', 'địa phương': '📍',
    'giới trẻ': '🌈', 'bất động sản': '🏠', 'giáo dục': '🎓', 'số hóa': '🔢', 'người lính': '🎖️',
    'nhịp sống phương nam': '🏙️', 'xã hội': '👥', 'quốc tế': '🌐', 'y tế': '💉', 'địa ốc': '🏗️',
    'đô thị': '🌆', 'công nghệ': '💻', 'khoa học công nghệ': '🚀', 'nhà đất': '🏡', 
    'giáo dục - hướng nghiệp': '📖', 'bạn đọc làm báo': '✍️'
}

# --- HÀM KHỞI TẠO HỆ THỐNG ---
@st.cache_data
def load_recommendations():
    """Lấy mỗi topic 1 câu claim ví dụ từ tập dữ liệu Master"""
    path = settings.DATA_PATHS.get("train")
    recs = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            random.shuffle(data)
            for item in data:
                topic = item.get("Topic", "khác").strip().lower()
                if topic not in recs:
                    recs[topic] = item.get("Statement", "")
    return recs

@st.cache_resource
def init_core_system():
    """Khởi tạo database và các module xử lý"""
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
                
    return ret_mod, EvidenceSelectionModule(db), url_map, VietnameseReranker()

ret_mod, ev_mod, url_to_context, reranker = init_core_system()
recs_dict = load_recommendations()

# Quản lý Session State cho ô nhập liệu
if "main_input" not in st.session_state:
    st.session_state["main_input"] = ""

# --- SIDEBAR: ĐIỀU KHIỂN & THAM SỐ ---
st.sidebar.title("🎮 Control Panel")
target_stage = st.sidebar.selectbox("Giai đoạn dừng xử lý:", 
                                    ["Document Retrieval", "Evidence Selection", "Claim Verification"])

# Tùy chỉnh hiển thị Grid
st.sidebar.subheader("🎨 Giao diện gợi ý")
grid_cols = st.sidebar.slider("Số cột hiển thị Topic:", 2, 8, 6)

# 1. Tham số Document Retrieval
with st.sidebar.expander("1. Document Retrieval Settings", expanded=True):
    dr_w_emb = st.slider("Embedding Weight", 0.0, 1.0, 0.4, key="dr_emb")
    dr_w_bm25 = st.slider("BM25 Weight", 0.0, 1.0, 0.3, key="dr_bm25")
    dr_w_tfidf = 1.0 - dr_w_emb - dr_w_bm25
    st.slider("TF-IDF Weight (Cố định)", 0.0, 1.0, max(0.0, dr_w_tfidf), disabled=True)
    
    if dr_w_emb + dr_w_bm25 > 1.0:
        st.error("Tổng trọng số vượt quá 1.0!")

    dr_use_rerank = st.toggle("Sử dụng Reranker cho Document?")
    dr_top_k = st.number_input("Top K URLs (before rerank)", 1, 10, 3 if dr_use_rerank else 1)

# 3. Tham số Claim Verification (Xác định chế độ để ẩn Step 2)
v_mode = "Selected Evidences"
if target_stage == "Claim Verification":
    with st.sidebar.expander("3. Claim Verification Settings", expanded=True):
        v_mode = st.radio("Xác thực dựa trên:", ["Full Context", "Selected Evidences"])
        plm_list = [
            "tranthaihoa/xlm_base_full", "tranthaihoa/xlm_large_full",
            "tranthaihoa/ViBERT_Full", "tranthaihoa/mBert_Full",
            "tranthaihoa/phobert_base_Context", "tranthaihoa/phobert_large_Context"
        ]
        selected_model = st.selectbox("Chọn Model PLM:", plm_list)

# 2. Tham số Evidence Selection (Ẩn nếu chọn Full Context)
show_ev = (target_stage == "Evidence Selection") or (target_stage == "Claim Verification" and v_mode == "Selected Evidences")
if show_ev:
    with st.sidebar.expander("2. Evidence Selection Settings", expanded=True):
        ev_w_emb = st.slider("Evid. Embedding Weight", 0.0, 1.0, 0.6, key="ev_emb")
        ev_w_bm25 = st.slider("Evid. BM25 Weight", 0.0, 1.0, 0.2, key="ev_bm25")
        ev_w_tfidf = 1.0 - ev_w_emb - ev_w_bm25
        st.slider("Evid. TF-IDF (Cố định)", 0.0, 1.0, max(0.0, ev_w_tfidf), disabled=True)
        
        ev_use_rerank = st.toggle("Sử dụng Reranker cho Evidence?", value=True)
        
        # POPUP THÔNG TIN CƠ SỞ RERANK THEO YÊU CẦU
        if ev_use_rerank:
            st.info("""
            **ℹ️ Cơ chế Rerank Phân cấp (Hierarchical):**
            1. Hệ thống lấy **Top K** ứng viên ban đầu.
            2. Nếu có bằng chứng đạt điểm tín nhiệm > **T1**, lấy tất cả các bằng chứng đó.
            3. Ngược lại, hệ thống sắp xếp giảm dần và lấy bằng chứng đầu tiên. Các bằng chứng tiếp theo sẽ được chọn nếu khoảng cách điểm so với thằng liền trước nhỏ hơn **T2**.
            """)
            ev_top_k_input = st.number_input("Số lượng bằng chứng lấy ra trước khi Rerank:", 3, 20, 10)
            t1 = st.slider("Confidence Threshold (T1)", 0.6, 1.0, 0.75)
            t2 = st.slider("Gap Threshold (T2)", 0.0, 0.15, 0.05)
        else:
            ev_top_k_input = st.number_input("Số lượng bằng chứng (Top K):", 1, 10, 3)

# --- KHU VỰC GỢI Ý (CUSTOMIZABLE GRID) ---
st.subheader("💡 Gợi ý Claim theo chủ đề")
topic_list = list(recs_dict.keys())

for i in range(0, len(topic_list), grid_cols):
    cols = st.columns(grid_cols)
    for j in range(grid_cols):
        if i + j < len(topic_list):
            topic = topic_list[i + j]
            icon = TOPIC_ICONS.get(topic, '📝')
            if cols[j].button(f"{icon} {topic.capitalize()}", key=f"btn_{topic}"):
                st.session_state["main_input"] = recs_dict[topic]
                st.rerun()

st.divider()

# --- GIAO DIỆN CHÍNH ---
claim_text = st.text_area("Nhập nội dung cần kiểm chứng (Claim):", 
                          key="main_input", height=120)

if st.button("🚀 Bắt đầu thực hiện xử lý", type="primary"):
    if not claim_text.strip():
        st.warning("Vui lòng nhập nội dung!")
        st.stop()

    # BƯỚC 1: DOCUMENT RETRIEVAL
    with st.status("🔍 Đang truy xuất bài báo liên quan...") as s:
        dr_weights = (dr_w_bm25, 1.0 - dr_w_emb - dr_w_bm25, dr_w_emb)
        urls = ret_mod.get_top_k_url(claim_text, top_k=dr_top_k, weights=dr_weights)
        
        if dr_use_rerank:
            class Item:
                def __init__(self, url, content): 
                    self.url = url
                    self.page_content = content
            cands = [Item(u, url_to_context[u]) for u in urls]
            best_url = reranker.rerank(claim_text, cands)[0]['document'].url
        else:
            best_url = urls[0]
        s.update(label="✅ Đã tìm thấy bài báo nguồn!", state="complete")

    st.markdown(f"**Nguồn:** [{best_url}]({best_url})")
    full_text = url_to_context.get(best_url, "")

    if target_stage == "Document Retrieval":
        st.subheader("Nội dung bài báo:")
        st.write(full_text)
        st.stop()

    # BƯỚC 2: EVIDENCE SELECTION
    selected_evidences = []
    if show_ev:
        with st.status("📍 Đang trích xuất bằng chứng xác thực...") as s:
            ev_weights = (ev_w_bm25, 1.0 - ev_w_emb - ev_w_bm25, ev_w_emb)
            
            if not ev_use_rerank:
                selected_evidences = ev_mod.select_top_k_evidence(claim_text, best_url, top_k=ev_top_k_input, weights=ev_weights)
            else:
                cands = ev_mod.select_top_k_evidence(claim_text, best_url, top_k=ev_top_k_input, weights=ev_weights)
                reranked_ev = reranker.rerank(claim_text, cands)
                
                # Rule 1: Threshold T1
                high_score_entries = [res for res in reranked_ev if res['rerank_score'] > t1]
                if high_score_entries:
                    selected_evidences = [res['document'] for res in high_score_entries]
                else:
                    # Rule 2: Hierarchy Gap T2
                    selected_evidences = [reranked_ev[0]['document']]
                    for i in range(1, len(reranked_ev)):
                        if (reranked_ev[i-1]['rerank_score'] - reranked_ev[i]['rerank_score']) < t2:
                            selected_evidences.append(reranked_ev[i]['document'])
                        else: break
            s.update(label=f"✅ Đã trích xuất {len(selected_evidences)} bằng chứng!", state="complete")

        # Hiển thị Highlight
        highlighted_html = full_text
        for ev in selected_evidences:
            snippet = ev.page_content.strip()
            highlighted_html = highlighted_html.replace(snippet, f'<span class="highlight">{snippet}</span>')
        
        st.subheader("Minh chứng trực quan:")
        st.markdown(f"<div style='text-align: justify;'>{highlighted_html}</div>", unsafe_allow_html=True)
    else:
        st.subheader("Nội dung bài báo (Chế độ Full Context):")
        st.write(full_text)

    if target_stage == "Evidence Selection":
        st.stop()

    # BƯỚC 3: CLAIM VERIFICATION
    if target_stage == "Claim Verification":
        with st.spinner("⚖️ Đang tiến hành xác thực claim..."):
            verifier = ClaimVerificationModule(selected_model)
            result = verifier.verify_claim(
                claim_text, 
                full_context=full_text if v_mode == "Full Context" else None,
                evidences=selected_evidences if v_mode == "Selected Evidences" else None
            )
            
            st.divider()
            st.subheader("🏁 Kết quả xác thực:")
            label = result['label_name']
            if label == "Supported":
                st.success("✅ **CHÍNH XÁC**: Nội dung khớp với bài báo.")
            elif label == "Refuted":
                st.error("❌ **SAI SỰ THẬT**: Nội dung mâu thuẫn với bài báo.")
            else:
                st.warning("❓ **KHÔNG ĐỦ THÔNG TIN**: Không đủ dữ liệu để kết luận.")