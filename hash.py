import hashlib
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def generate_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def main():
    Tk().withdraw()  # Hide the root window
    file_path = askopenfilename(title="Select a file to hash")
    if file_path:
        hash_result = generate_sha256(file_path)
        print(f"SHA-256 hash of {file_path}:\n{hash_result}")
    else:
        print("No file selected.")

if __name__ == "__main__":
    main()
