import datasets

try:
    ds = datasets.load_dataset('QIAIUNCC/EYE-lit-complete', split='train', streaming=True)
    for i, row in enumerate(ds):
        print(f"\\n--- ROW {i} ---")
        print("KEYS:", row.keys())
        content = str(row.get('page_content', ''))
        print(f"page_content length: {len(content)}")
        print(f"page_content snippet: {content[:200]}")
        if i >= 2:
            break
except Exception as e:
    print("Error:", e)
