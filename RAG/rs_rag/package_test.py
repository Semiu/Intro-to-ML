import nltk

print(nltk.data.path)

try:
    nltk.data.find('tokenizers/punkt_tab')
    print("✅ 'punkt_tab' is already downloaded.")
except LookupError:
    print("❌ 'punkt_tab' is NOT downloaded.")
    #nltk.download("punkt_tab")
    #'/home/ec2-user/anaconda3/lib/nltk_data', '/usr/share/nltk_data', '/usr/local/share/nltk_data', '/usr/lib/nltk_data', '/usr/local/lib/nltk_data']
