from evaluate import load
rouge = load("rouge")
print(rouge.compute(predictions=["hello world"], references=["hello world"]))
# 应输出类似 {'rouge1': ..., 'rouge2': ..., 'rougeL': ...}