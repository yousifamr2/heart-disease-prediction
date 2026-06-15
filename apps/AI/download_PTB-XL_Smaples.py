import wfdb
import os
import urllib.request

# اسم الفولدر اللي هتنزل فيه الداتا
download_dir = './ptbxl_sample'
os.makedirs(download_dir, exist_ok=True)

# 1. تحميل ملفات الـ CSV
print("Downloading CSV files...")
base_url = "https://physionet.org/files/ptb-xl/1.0.3/"
csv_files = ['ptbxl_database.csv', 'scp_statements.csv']

for file_name in csv_files:
    file_url = base_url + file_name
    save_path = os.path.join(download_dir, file_name)
    # لو الملف موجود مفيش داعي يحمله تاني
    if not os.path.exists(save_path):
        print(f"Downloading {file_name}...")
        urllib.request.urlretrieve(file_url, save_path)
    else:
        print(f"{file_name} already exists. Skipping.")

# 2. الحصول على لستة بكل الملفات من سيرفر PhysioNet
print("Fetching the complete record list from PhysioNet...")
all_records = wfdb.get_record_list('ptb-xl')

# 3. فلترة اللستة عشان نختار ملفات المجلد الأول (00000) بتردد 100Hz فقط
folder_00000_records = [record for record in all_records if record.startswith('records100/00000/')]

print(f"Found {len(folder_00000_records)} records in the first folder.")
print("Starting download... (This might take a few minutes depending on your internet speed)")

# 4. تحميل المجلد بالكامل
wfdb.dl_database('ptb-xl', dl_dir=download_dir, records=folder_00000_records)

print("Download complete! Check the 'ptbxl_sample' folder.")