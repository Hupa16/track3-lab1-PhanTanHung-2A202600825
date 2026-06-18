import json
import os

def generate_data():
    input_path = "data/hotpot_mini.json"
    output_path = "data/test_set_100.json"
    
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    new_data = []
    # Lặp lại dữ liệu để đạt 100 mẫu (8 mẫu gốc -> 104 mẫu)
    for i in range(13):
        for item in data:
            new_item = item.copy()
            new_item["qid"] = f"{item['qid']}_{i}"
            new_data.append(new_item)
            
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=4, ensure_ascii=False)
        
    print(f"Generated {len(new_data)} samples at {output_path}")

if __name__ == "__main__":
    generate_data()
