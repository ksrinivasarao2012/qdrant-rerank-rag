import os
import re
import xml.etree.ElementTree as ET
import json
from pathlib import Path

# Project directories, resolved relative to this file so the script works on any
# machine (local, HF Spaces, CI) rather than only on D:\
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Main data files
POSTS_XML_PATH = RAW_DIR / "Posts.xml"
OUTPUT_JSONL_PATH = PROCESSED_DIR / "posts.jsonl"

def clean_html(raw_html: str) -> str:
    """
    Cleans raw HTML text from StackExchange posts.
    
    1. Removes all HTML tags (like <p>, <code>) using regular expressions.
    2. Converts XML entities (like &lt; and &gt;) back to plain characters (< and >).
    """
    if not raw_html:
        return ""
    
    # Strip HTML tags
    clean_text = re.sub(r'<[^>]+>', '', raw_html)
    
    # Replace common HTML code symbols with actual text characters
    clean_text = (clean_text
                  .replace("&lt;", "<")
                  .replace("&gt;", ">")
                  .replace("&amp;", "&")
                  .replace("&quot;", '"')
                  .replace("&#39;", "'"))
    
    return clean_text.strip()



def parse_tags(tags_str: str) -> list:
    """
    Extracts tags from StackExchange format (either <tag1><tag2> or |tag1|tag2|).
    """
    if not tags_str:
        return []
    if "|" in tags_str:
        return [t.strip() for t in tags_str.split("|") if t.strip()]
    return re.findall(r'<([^>]+)>', tags_str)

def main():
    # Verify raw data XML dump is present before starting
    if not POSTS_XML_PATH.exists():
        print(f"Error: Raw dataset not found at {POSTS_XML_PATH.resolve()}")
        print("Please download and place the XML dump in data/raw/ before running this parser.")
        return

    # Ensure output directory exists
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Pass 1: Reading questions metadata into memory...")
    # Stores question metadata in memory for quick mapping.
    # Map structure: { question_id: {title, accepted_answer_id, score, tags} }
    questions = {}
    
    # iterparse's .root is None until parsing starts, so we take the root from the
    # first 'start' event. Clearing the root each row is what actually frees memory:
    # elem.clear() empties a row but leaves it attached to root, so root's child
    # list would otherwise grow to millions of entries on a multi-GB file.
    context = ET.iterparse(str(POSTS_XML_PATH), events=("start", "end"))
    _, root = next(context)
    count = 0

    for event, elem in context:
        if event != "end" or elem.tag != "row":
            continue

        post_type = elem.attrib.get("PostTypeId")

        # PostTypeId "1" represents a Question
        if post_type == "1":
            q_id = int(elem.attrib.get("Id"))
            score = int(elem.attrib.get("Score", 0))
            title = elem.attrib.get("Title", "")
            accepted_ans_id = elem.attrib.get("AcceptedAnswerId")
            tags_str = elem.attrib.get("Tags", "")

            questions[q_id] = {
                "title": title,
                # Body is needed later to build the evaluation set (query = title + body).
                # Capturing it now avoids re-parsing the whole dump at Step 4.
                "body": clean_html(elem.attrib.get("Body", "")),
                "accepted_answer_id": int(accepted_ans_id) if accepted_ans_id else None,
                "score": score,
                "tags": parse_tags(tags_str),
                "created": elem.attrib.get("CreationDate", "")
            }
        count += 1
        if count % 100000 == 0:
            print(f"  Processed {count} rows...")

        elem.clear()
        root.clear()

    print(f"Finished Pass 1. Found {len(questions)} total questions.")

    # Filter questions to keep all high-quality questions (score >= 1) that have an accepted answer
    print("Filtering questions to keep those with score >= 1 and an accepted answer...")
    top_questions = {
        q_id: q_data for q_id, q_data in questions.items()
        if q_data["score"] >= 1 and q_data["accepted_answer_id"] is not None
    }
    
    # Store question IDs in a set for O(1) instant lookup times
    top_question_ids = set(top_questions.keys())

    print(f"Pass 2: Extracting answers linked to the {len(top_question_ids)} selected questions...")
    # Reset XML stream iterator for the second pass
    context = ET.iterparse(str(POSTS_XML_PATH), events=("start", "end"))
    _, root = next(context)
    ans_count = 0

    # Write output to JSONL (one JSON object per line) to keep file reads/writes memory-friendly
    with open(OUTPUT_JSONL_PATH, "w", encoding="utf-8") as f:
        for event, elem in context:
            if event != "end" or elem.tag != "row":
                continue

            # PostTypeId "2" represents an Answer
            if elem.attrib.get("PostTypeId") == "2":
                parent_id = int(elem.attrib.get("ParentId", 0))

                # Only index the answer if it belongs to one of our selected
                # questions (score >= 1 and has an accepted answer)
                if parent_id in top_question_ids:
                    ans_id = int(elem.attrib.get("Id"))
                    score = int(elem.attrib.get("Score", 0))
                    body_html = elem.attrib.get("Body", "")

                    parent_q = top_questions[parent_id]
                    # Check if this answer was accepted (marked correct) by the original asker
                    is_accepted = (ans_id == parent_q["accepted_answer_id"])

                    # Build a clean structured record for the RAG database
                    record = {
                        "answer_id": ans_id,
                        "question_id": parent_id,
                        "question_title": parent_q["title"],
                        "question_body": parent_q["body"],
                        "answer_text": clean_html(body_html),
                        "score": score,
                        "is_accepted": is_accepted,
                        "tags": parent_q["tags"],
                        "created": elem.attrib.get("CreationDate", ""),
                        "url": f"https://stats.stackexchange.com/a/{ans_id}"
                    }
                    f.write(json.dumps(record) + "\n")
                    ans_count += 1

            elem.clear()
            root.clear()

    print(f"Finished Pass 2. Extracted {ans_count} answers to {OUTPUT_JSONL_PATH.resolve()}")

if __name__ == "__main__":
    main()
