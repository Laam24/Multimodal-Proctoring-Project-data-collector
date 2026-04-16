import os
import re
import csv

BASE_OUTPUT_DIR = "Honeypot_Sessions"
OUTPUT_CSV = "master_labels.csv"

# Keyword Dictionaries to map your random phrases to strict model labels
KEYWORDS_NORMAL =["honest", "honestly", "fair", "normal", "didn't cheat", "did not cheat"]
KEYWORDS_PHYSICAL =["phone", "hand", "side", "lap", "note", "book", "paper", "looked away"]
KEYWORDS_DIGITAL =["chrome", "tab", "browser", "laptop", "screen", "chatgpt", "gpt", "google", "search"]

def get_standard_label(text):
    text_lower = text.lower()
    
    # Check for Normal behavior first
    if any(word in text_lower for word in KEYWORDS_NORMAL):
        return "NORMAL"
        
    # Check for Digital Cheating
    if any(word in text_lower for word in KEYWORDS_DIGITAL):
        return "CHEAT_DIGITAL"
        
    # Check for Physical Cheating
    if any(word in text_lower for word in KEYWORDS_PHYSICAL):
        return "CHEAT_PHYSICAL"
        
    # If the script doesn't recognize the words you used, it flags it for your manual review
    return "UNKNOWN_PLEASE_FIX"

def main():
    print("Scanning session folders for log.txt files...\n")
    
    master_data =[]
    
    if not os.path.exists(BASE_OUTPUT_DIR):
        print(f"Error: Could not find folder {BASE_OUTPUT_DIR}")
        return

    # Loop through every session folder
    for session_folder in os.listdir(BASE_OUTPUT_DIR):
        session_path = os.path.join(BASE_OUTPUT_DIR, session_folder)
        
        if os.path.isdir(session_path):
            log_file = os.path.join(session_path, "log.txt")
            
            if os.path.exists(log_file):
                with open(log_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    
                    for line in lines:
                        line = line.strip()
                        if not line: continue
                        
                        # Find the question number (e.g., "Q1", "Q3")
                        q_match = re.search(r'(Q[1-5])', line, re.IGNORECASE)
                        q_num = q_match.group(1).upper() if q_match else "UNKNOWN_Q"
                        
                        # Get the standardized label
                        std_label = get_standard_label(line)
                        
                        master_data.append([session_folder, q_num, std_label, line])

    # Write the compiled data to a Master CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Session_Folder", "Question", "Standard_Label", "Original_Log_Text"])
        writer.writerows(master_data)

    print(f"Successfully processed {len(master_data)} log entries.")
    print(f"Saved to {OUTPUT_CSV}. Please open this file and review it!")

if __name__ == "__main__":
    main()