import streamlit as st
import pandas as pd
from src.charts import get_top_items
from src.search import build_search_index, search_documents, get_related_documents
from src.processing import generate_summary, extract_keywords, extract_entities


st.set_page_config(page_title="DocDB-News", layout="wide")


# -----------------------
# Session State
# -----------------------
if "processed_df" not in st.session_state:
    st.session_state.processed_df = None


# -----------------------
# Title
# -----------------------
st.title("DocDB-News")
st.write("Analyze uploaded news datasets with a simplified DocDB-style interface.")


# -----------------------
# Upload
# -----------------------
uploaded_file = st.file_uploader("Upload a news CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)


    # -----------------------
    # Column Mapping
    # -----------------------
    st.subheader("Column Mapping")

    columns = [col for col in df.columns if col.lower() != "guid"]
    none_option = ["None"] + columns

    title_col = st.selectbox(
        "Title Column",
        none_option,
        index=none_option.index("title") if "title" in none_option else 0
    )

    text_col = st.selectbox(
        "Text Column",
        columns,
        index=columns.index("description") if "description" in columns else 0
    )

    date_col = st.selectbox(
        "Date Column (optional)",
        none_option,
        index=none_option.index("pubDate") if "pubDate" in none_option else 0
    )

    link_col = st.selectbox(
        "Link Column (optional)",
        none_option,
        index=none_option.index("link") if "link" in none_option else 0
    )


    # -----------------------
    # Processing
    # -----------------------
    if st.button("Process Dataset"):
        with st.spinner("Processing dataset..."):

            if title_col != "None":
                title_series = df[title_col].fillna("").astype(str)
            else:
                title_series = pd.Series(["Untitled"] * len(df), index=df.index)

            text_series = df[text_col].fillna("").astype(str)

            date_series = (
                df[date_col].fillna("").astype(str)
                if date_col != "None"
                else pd.Series([""] * len(df), index=df.index)
            )
            link_series = (
                df[link_col].fillna("").astype(str)
                if link_col != "None"
                else pd.Series([""] * len(df), index=df.index)
            )

            processed_df = pd.DataFrame(
                {
                    "title": title_series,
                    "text": text_series,
                    "date": date_series,
                    "link": link_series
                }
            )

            processed_df = processed_df[processed_df["text"] != ""].copy()

            processed_df = processed_df.drop_duplicates(
                subset=["title", "text"]
            ).reset_index(drop=True)

            processed_df["doc_id"] = range(1, len(processed_df) + 1)
            processed_df["summary"] = processed_df["text"].apply(generate_summary)
            processed_df["keywords"] = processed_df["text"].apply(extract_keywords)
            processed_df["entities"] = processed_df["text"].apply(extract_entities)

            processed_df["keywords"] = processed_df["keywords"].apply(
                lambda x: ", ".join(x)
            )

            processed_df["entities"] = processed_df["entities"].apply(
                lambda x: ", ".join(x)
            )

            processed_df = processed_df[
                [
                    "doc_id",
                    "title",
                    "text",
                    "date",
                    "link",
                    "summary",
                    "keywords",
                    "entities"
                ]
            ]

            st.session_state.processed_df = processed_df

        st.success("Processing complete!")


    # -----------------------
    # After Processing
    # -----------------------
    if st.session_state.processed_df is not None:
        processed_df = st.session_state.processed_df


        # -----------------------
        # Filters
        # -----------------------
        st.subheader("Filters")

        col1, col2 = st.columns(2)

        with col1:
            keyword_input = st.text_input("Filter by Keyword")

        with col2:
            entity_input = st.text_input("Filter by Entity")

        filtered_df = processed_df.copy()

        if keyword_input:
            filtered_df = filtered_df[
                filtered_df["keywords"].str.contains(
                    keyword_input, case=False, na=False
                )
            ]

        if entity_input:
            filtered_df = filtered_df[
                filtered_df["entities"].str.contains(
                    entity_input, case=False, na=False
                )
            ]

        if not filtered_df.empty:
            vectorizer, doc_matrix = build_search_index(filtered_df)
        else:
            vectorizer, doc_matrix = None, None


        # -----------------------
        # Stats
        # -----------------------
        st.subheader("Dataset Statistics")

        unique_keywords = set()
        for kw_list in filtered_df["keywords"]:
            for kw in str(kw_list).split(","):
                kw = kw.strip()
                if kw:
                    unique_keywords.add(kw)

        unique_entities = set()
        for ent_list in filtered_df["entities"]:
            for ent in str(ent_list).split(","):
                ent = ent.strip()
                if ent:
                    unique_entities.add(ent)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Docs", len(processed_df))
        col2.metric("Visible Docs", len(filtered_df))
        col3.metric("Unique Keywords", len(unique_keywords))
        col4.metric("Unique Entities", len(unique_entities))


        # -----------------------
        # Table
        # -----------------------
        st.subheader("Processed Dataset")

        display_columns = st.multiselect(
            "Choose columns to display",
            options=filtered_df.columns.tolist(),
            default=[
                "doc_id",
                "title",
                "date",
                "summary",
                "keywords",
                "entities"
            ],
        )

        if display_columns:
            max_display = min(50, len(filtered_df))
            default_display = min(10, max_display)

            display_count = st.slider(
                "Number of rows to display",
                min_value=5,
                max_value=max_display,
                value=default_display,
                step=5
            )

            st.caption(f"Showing top {display_count} of {len(filtered_df)} rows")

            st.dataframe(
                filtered_df[display_columns].head(display_count),
                use_container_width=True,
            )
        else:
            st.info("Select at least one column to display.")

        export_df = filtered_df.drop(columns=["text"], errors="ignore")

        csv = export_df.to_csv(index=False).encode("utf-8")
        
        st.download_button(
            "Download CSV",
            csv,
            "processed.csv",
            "text/csv"
        )


        # -----------------------
        # Charts
        # -----------------------
        st.subheader("Analysis")

        top_keywords_df = get_top_items(filtered_df["keywords"])
        top_entities_df = get_top_items(filtered_df["entities"])

        col1, col2 = st.columns(2)

        with col1:
            st.write("Top Keywords")
            if not top_keywords_df.empty:
                st.bar_chart(top_keywords_df.set_index("item")["count"])
            else:
                st.info("No keyword data available.")

        with col2:
            st.write("Top Entities")
            if not top_entities_df.empty:
                st.bar_chart(top_entities_df.set_index("item")["count"])
            else:
                st.info("No entity data available.")


        # -----------------------
        # Search
        # -----------------------
        st.subheader("Search Documents")
        query = st.text_input("Enter a search query", value="")

        if query.strip():
            results_df = search_documents(
                filtered_df,
                query,
                vectorizer,
                doc_matrix
            )

            if not results_df.empty:
                max_display = min(50, len(results_df))
                default_display = min(10, max_display)

                display_count = st.slider(
                    "Number of search results to show",
                    min_value=5,
                    max_value=max_display,
                    value=default_display,
                    step=5
                )

                st.write(
                    f"Found {len(results_df)} matching documents. Showing top {display_count}."
                )

                st.dataframe(
                    results_df[["score", "title", "summary"]].head(display_count),
                    use_container_width=True
                )

                results_export_df = results_df.drop(columns=["text"], errors="ignore")
                results_csv = results_export_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Search Results CSV",
                    results_csv,
                    "search_results.csv",
                    "text/csv"
                )
            else:
                st.warning("No results found.")


        # -----------------------
        # Document Viewer
        # -----------------------
        st.subheader("Document Viewer")

        if filtered_df.empty:
            st.warning("No documents available for the current filters.")
        else:
            viewer_options = {
                f"{row.doc_id} - {row.title}": idx
                for idx, row in filtered_df[["doc_id", "title"]].iterrows()
            }

            selected_label = st.selectbox(
                "Select Document",
                list(viewer_options.keys())
            )
            selected_index = viewer_options[selected_label]
            selected_doc = filtered_df.loc[selected_index]

            if selected_doc["link"]:
                st.link_button("Open Article", selected_doc["link"])

            st.write("**Title:**")
            st.write(selected_doc["title"])
            
            st.write("**Document ID:**")
            st.write(selected_doc["doc_id"])

            st.write("**Date:**")
            st.write(selected_doc["date"])

            st.write("**Summary:**")
            st.write(selected_doc["summary"])

            st.write("**Keywords:**")
            st.write(selected_doc["keywords"])

            st.write("**Entities:**")
            st.write(selected_doc["entities"])

            st.write("### Related Documents")
            related_df = get_related_documents(
                filtered_df,
                selected_index,
                vectorizer,
                doc_matrix
            )

            if not related_df.empty:
                st.dataframe(
                    related_df[["similarity", "title", "summary"]].head(5),
                    use_container_width=True
                )
            else:
                st.info("No related documents available.")

else:
    st.info("Upload a CSV file to begin.")
