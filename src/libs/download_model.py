from transformers import BertTokenizer, BertForSequenceClassification

# Tải mô hình và tokenizer trước
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='./models')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', cache_dir='./models')
