
CHUNK_SIZE = 40
MODEL_ID = "answerdotai/ModernBERT-base"

PUNCT_SYMBOLS = {".": "PERIOD", ",": "COMMA", "?": "QUESTION"}
punct_label2id = {
    "O": 0,
    ",": 1,
    ".": 2,
    "?": 3
}
punct_id2label = {v: k for k, v in punct_label2id.items()}

punct_id2symbol = {0: "", 1: ",", 2: ".", 3: "?"}

cap_label2id = {"NO_CAP": 0, "CAP": 1}
cap_id2label = {v: k for k, v in cap_label2id.items()}


inference_text = [
    "hello how are you doing today i hope everything is fine",
    "this is an example of a model that predicts punctuation and capitalization",
    "what time is the meeting scheduled for tomorrow",
    "can you explain how punctuation affects readability"]