
import sqlite3
from src.ingestion.filename_parser import parse_cell_guide_filename
from src.storage.normalize_speaker import normalize_speaker

DB_PATH = "data/sermons.db"

def fix_date_speakers():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Find all sermons where speaker contains "Date" (case insensitive)
    cursor.execute("SELECT sermon_id, ng_file, speaker FROM sermons WHERE speaker LIKE '%date%'")
    rows = cursor.fetchall()
    
    if not rows:
        print("No 'Date' speakers found.")
        return

    print(f"Found {len(rows)} problematic speakers. Attempting to fix...")
    
    for sermon_id, ng_file, old_speaker in rows:
        if not ng_file:
            continue
            
        parsed = parse_cell_guide_filename(ng_file)
        new_speaker = parsed.get("speaker")
        
        if new_speaker:
            normalized = normalize_speaker(new_speaker)
            if normalized and "Date" not in normalized:
                print(f"Updating {sermon_id}: '{old_speaker}' -> '{normalized}'")
                cursor.execute("UPDATE sermons SET speaker = ? WHERE sermon_id = ?", (normalized, sermon_id))
            else:
                print(f"Could not normalize speaker for {sermon_id} ({new_speaker})")
        else:
            print(f"No speaker found in filename for {sermon_id} ({ng_file})")
            
    # Hardcoded fixes for cases where filename doesn't help
    hardcoded_fixes = {
        "English_2015_14th-15th-Feb-2015-Celebrate-the-Lord_Notes.pdf": "Ps Jason Teo",
        "English_2015_3-4-Oct-2015-Components-of-a-Godly-Decision_Notes.pdf": "Ps Michael Ross Watson",
        "English_2015_4-5-April-2015-The-Fear-Factor-by-Mr-J_Notes-1.pdf": "Ps Jeffrey Aw",
        "English_2015_21-22-March-2015-Forgive-as-the-Lord_notes.pdf": "Ps Ernest Chow",
        "English_2015_19-July-2015-The-Works-of-The-Spirit_Notes.pdf": "Dr Ian Jagelman",
        "English_2015_25-26-July-15-This-Life-the-Next-by-Ps-Chew-Weng-Chee_Members-Guide.pdf": "Ps Chew Weng Chee",
        "English_2015_07-08-November-2015-We-are-One-Body-by-by-eGHC-Members-Guide.pdf": "Elder Goh Hock Chye",
        "English_2015_11-12-July-2015-Prayer-in-the-Belly-of-the-Fish-by-Edric-Sng-Members27-Guide.pdf": "Ps Edric Sng",
        "English_2017_2-3-Dec-2017-Perilous-Times-by-DSP-Chua-Seng-Lee-Members-Guide.pdf": "SP Chua Seng Lee",
    }
    
    for filename, speaker in hardcoded_fixes.items():
        normalized = normalize_speaker(speaker)
        print(f"Applying hardcoded fix for {filename}: {normalized}")
        cursor.execute("UPDATE sermons SET speaker = ? WHERE ng_file = ?", (normalized, filename))

    conn.commit()
    conn.close()
    print("Fix complete.")

if __name__ == "__main__":
    fix_date_speakers()
