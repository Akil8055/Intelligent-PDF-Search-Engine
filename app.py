import os
import tempfile
from typing import Optional

import streamlit as st
import streamlit.components.v1 as components
import threading
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from functools import partial
from urllib.parse import quote

from core.document_manager import add_document, delete_document, list_documents
from core.search import search
from core.embedder import Embedder
from core.indexer import VectorIndexer
from utils.storage import load_index, load_metadata, ensure_directories
from utils.config import UPLOAD_DIR, TOP_K, EMBEDDING_DIM


st.set_page_config(page_title="Intelligent Document Search", page_icon="🔍", layout="wide")


@st.cache_resource
def get_embedder():
    return Embedder()


@st.cache_resource
def get_indexer():
    # Create indexer and attempt to populate with existing index & metadata
    indexer = VectorIndexer(embedding_dim=EMBEDDING_DIM)

    # Try to load FAISS index and metadata (from utils.storage)
    try:
        ensure_directories()
        loaded = load_index(EMBEDDING_DIM)
        metadata = load_metadata()
        # Only assign if loaded index has non-zero size or metadata present
        if hasattr(loaded, "ntotal") and loaded.ntotal > 0:
            indexer.index = loaded
        if metadata:
            indexer.metadata = metadata
    except Exception:
        # Fallback: keep fresh indexer
        pass

    return indexer


def shorten(text: str, length: int = 200) -> str:
    if not text:
        return ""
    s = text.strip().replace("\n", " ")
    return s if len(s) <= length else s[: length - 1].rstrip() + "…"


def main():
    ensure_directories()

    st.title("📚 Intelligent Document Search Engine")
    st.write("Upload PDF files, index them, and perform semantic search across their contents.")

    embedder = get_embedder()
    indexer = get_indexer()
    # Start a lightweight HTTP server to serve uploaded PDFs for reliable iframe previews
    if "file_server_url" not in st.session_state:
        def start_file_server(start_port: int = 8502, max_port: int = 8512):
            abs_upload_dir = os.path.abspath(UPLOAD_DIR)
            for port in range(start_port, max_port):
                try:
                    handler = partial(SimpleHTTPRequestHandler, directory=abs_upload_dir)
                    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)

                    def serve():
                        try:
                            httpd.serve_forever()
                        except Exception:
                            pass

                    thread = threading.Thread(target=serve, daemon=True)
                    thread.start()
                    return f"http://127.0.0.1:{port}", abs_upload_dir
                except OSError:
                    continue
            return None, abs_upload_dir

        url, served_dir = start_file_server()
        st.session_state.file_server_url = url
        st.session_state.file_server_dir = served_dir
        if url:
            st.session_state.file_server_running = True
        else:
            st.session_state.file_server_running = False
    # initialize session_state containers
    if "last_results" not in st.session_state:
        st.session_state.last_results = []

    # Use Streamlit's sidebar for a closable document manager
    with st.sidebar:
        st.header("🗂️ Documents")

        # File server status for PDF preview
        if st.session_state.get("file_server_running"):
            fs_url = st.session_state.get("file_server_url")
            st.caption(f"PDF preview server: {fs_url}")
        else:
            st.caption("PDF preview server not running — inline previews may be blocked by browsers.")

        # Show uploaded/indexed documents first
        st.subheader("Uploaded Documents")
        docs = list_documents(indexer)
        if not docs:
            st.info("No documents indexed yet.")
        else:
            for name in docs:
                cols = st.columns([4, 1])
                cols[0].write(name)
                # use a compact trash icon button instead of text
                if cols[1].button("🗑️", key=f"del_{name}", help="Delete document"):
                    ok = delete_document(name, indexer)
                    if ok:
                        st.success(f"Deleted {name} (marked inactive).")
                    else:
                        st.info(f"No active chunks found for {name}.")

        st.markdown("---")

        # File uploader below the document list; uploads are processed immediately
        uploaded = st.file_uploader("Upload PDF", type=["pdf"], accept_multiple_files=True)
        if uploaded:
            added = 0
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            processed = set(st.session_state.get("processed_uploads", []))
            for f in uploaded:
                key = f"{f.name}:{len(f.getvalue())}"
                if key in processed:
                    continue

                dest_name = f.name
                dest_path = os.path.join(UPLOAD_DIR, dest_name)

                if os.path.exists(dest_path):
                    base, ext = os.path.splitext(dest_name)
                    counter = 1
                    while True:
                        candidate = f"{base}_{counter}{ext}"
                        candidate_path = os.path.join(UPLOAD_DIR, candidate)
                        if not os.path.exists(candidate_path):
                            dest_name = candidate
                            dest_path = candidate_path
                            break
                        counter += 1

                with open(dest_path, "wb") as out:
                    out.write(f.getvalue())

                try:
                    add_document(dest_path, embedder, indexer)
                    added += 1
                    processed.add(key)
                except Exception as e:
                    st.error(f"Failed to add {f.name}: {e}")

            st.session_state["processed_uploads"] = list(processed)
            if added:
                st.success(f"Added {added} document(s).")

    # -----------------
    # Center: Search
    # -----------------
    with st.container():
        st.header("🔎 Search")
        query = st.text_input("Search query", value="")
        top_k = TOP_K

        search_clicked = st.button("Search")

        results = []
        if search_clicked:
            if not query or not query.strip():
                st.warning("Please enter a query.")
            else:
                if getattr(indexer.index, "ntotal", 0) == 0:
                    st.info("Index empty — upload documents first.")
                else:
                    with st.spinner("Searching..."):
                        try:
                            results = search(query, embedder, indexer, top_k=int(top_k))
                            # persist results across reruns so UI buttons don't clear them
                            st.session_state.last_results = results
                        except Exception as e:
                            st.error(f"Search failed: {e}")
        # prefer persisted results when available
        if not results:
            results = st.session_state.get("last_results", [])

        if results:
            st.subheader(f"Top {len(results)} Results")
            for i, r in enumerate(results, start=1):
                text = shorten(r.get("text", ""), 200)
                page = r.get("page", "?")
                source = r.get("source", "Unknown")
                score = r.get("score", 0.0)

                st.markdown(f"**{i}. {source}** — Page {page} — Score: {score:.4f}")
                st.write(text)

                if st.button("Open", key=f"open_{i}"):
                    # Try to open the PDF in a new tab (prefer the local file server).
                    server_url = st.session_state.get("file_server_url")
                    quoted_name = quote(source)
                    if server_url:
                        pdf_url = f"{server_url}/{quoted_name}#page={page}"
                        # open in new tab
                        components.html(f"<script>window.open('{pdf_url}', '_blank');</script>", height=0)
                    else:
                        # No file server: provide a Download button so the user can open the PDF locally
                        file_path_tmp = os.path.join(UPLOAD_DIR, source)
                        try:
                            with open(file_path_tmp, "rb") as fh:
                                st.download_button("Download PDF", data=fh.read(), file_name=source)
                        except Exception as e:
                            st.error(f"Could not prepare download: {e}")

    # Right-side PDF viewer removed; Open now opens in new tab or shows Download button.


if __name__ == "__main__":
    main()
