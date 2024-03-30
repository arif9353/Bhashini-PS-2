from langdetect import detect
def detect_language(text):
    try:
        language = detect(text)
        return language
    except:
        return "Language detection failed"

print(detect_language("বাংলা বাঙালি জাতির মাতৃভাষা।"))
