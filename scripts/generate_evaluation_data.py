import json
from pathlib import Path

def main():
    questions = []
    expected_pages = {}
    expected_answers = {}

    eval_dir = Path(__file__).resolve().parent.parent / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate 30 Comic Story Questions (story_comic.pdf)
    # Pages: 1-2 (Bear), 3 (Elephant)
    for i in range(1, 16):
        qid_bear = f"qb_{i}"
        questions.append({
            "id": qid_bear,
            "pdf": "story_comic.pdf",
            "question": f"Detail search item {i}: What does the brown bear checks hollow trees for in the big green forest?",
            "entities": ["bear", "forest", "honey"],
            "modality": "Text"
        })
        expected_pages[qid_bear] = [1, 2]
        expected_answers[qid_bear] = "Based on the document: This is the story of a brown bear. The bear lives in a big green forest. The bear likes to walk around looking for sweet things. Every morning, the bear checks the hollow trees to see if there is honey. Today, the bear is very lucky. The bear finds a large beehive in an old oak tree. The bear climbs the tree carefully. The bear eats the delicious honey. The bear feels extremely satisfied and goes to sleep under the tree."

    for i in range(1, 16):
        qid_ele = f"qe_{i}"
        questions.append({
            "id": qid_ele,
            "pdf": "story_comic.pdf",
            "question": f"Detail search item {i}: What is the large gray elephant looking for in the dry savanna?",
            "entities": ["elephant", "savanna", "water"],
            "modality": "Text"
        })
        expected_pages[qid_ele] = [3]
        expected_answers[qid_ele] = "Based on the document: This is the story of an elephant. The elephant is a very large gray animal with a long trunk. The elephant lives in the dry savanna. Today is very hot, so the elephant is looking for water. The elephant walks toward the blue river. The elephant drinks water and sprays it over its back."

    # 2. Generate 30 Financial Report Questions (financial_report.pdf)
    # Page 1: Revenue, EPS, Adjusted EPS table
    for i in range(1, 16):
        qid_rev = f"qf_rev_{i}"
        questions.append({
            "id": qid_rev,
            "pdf": "financial_report.pdf",
            "question": f"Financial inquiry {i}: What was the GAAP Operating Income in 2024?",
            "entities": ["GAAP", "2024", "EPS"],
            "modality": "Table"
        })
        expected_pages[qid_rev] = [1]
        expected_answers[qid_rev] = "Table indicates: | Year | Revenue | GAAP Operating Income | Non-GAAP Operating Income | EPS | Adjusted EPS | with data | 2024 | $1,000M | $200M | $250M | $2.00 | $2.50 |"

    for i in range(1, 16):
        qid_eps = f"qf_eps_{i}"
        questions.append({
            "id": qid_eps,
            "pdf": "financial_report.pdf",
            "question": f"Financial inquiry {i}: What was the Adjusted EPS difference between 2023 and 2024?",
            "entities": ["Adjusted", "EPS", "increase"],
            "modality": "Table"
        })
        expected_pages[qid_eps] = [1]
        expected_answers[qid_eps] = "The change in Adjusted EPS between 2023 and 2024 is 0.70 (based on structural table lookup: 2024=$2.50 and 2023=$1.80)."

    # 3. Generate 20 Negation/Hallucination Check Queries
    # Pages: story_comic.pdf (no lions or tigers)
    for i in range(1, 21):
        qid_neg = f"qn_{i}"
        questions.append({
            "id": qid_neg,
            "pdf": "story_comic.pdf",
            "question": f"Negation validation {i}: What does the wild lion eat in the savanna grasslands?",
            "entities": ["lion", "savanna"],
            "modality": "Text"
        })
        expected_pages[qid_neg] = []
        expected_answers[qid_neg] = "Insufficient document evidence to answer reliably."

    # Save to files
    with open(eval_dir / "questions.json", "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2)

    with open(eval_dir / "expected_pages.json", "w", encoding="utf-8") as f:
        json.dump(expected_pages, f, indent=2)

    with open(eval_dir / "expected_answers.json", "w", encoding="utf-8") as f:
        json.dump(expected_answers, f, indent=2)

    print(f"Generated 80-question evaluation dataset inside {eval_dir}.")

if __name__ == "__main__":
    main()
