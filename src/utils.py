import json
import ast


def cosine_similarity(vec1, vec2):
    """计算两个向量之间的余弦相似度"""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def parse_json(json_str, data_type="dict"):
    """解析 JSON 字符串并返回 Python 对象"""
    if data_type == "dict":
        # Find the first occurrence of '{' and the last occurrence of '}' to extract the JSON object
        start_idx = json_str.find('{')
        end_idx = json_str.rfind('}')
        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            json_str = json_str[start_idx:end_idx + 1]
        else:
            return None
    elif data_type == "list":
        # Find the first occurrence of '[' and the last occurrence of ']' to extract the JSON array
        start_idx = json_str.find('[')
        end_idx = json_str.rfind(']')
        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            json_str = json_str[start_idx:end_idx + 1]
        else:
            return None

    # 先尝试标准 JSON
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        # 再尝试 Python 字面量（支持单引号等）
        try:
            parsed = ast.literal_eval(json_str)
        except (ValueError, SyntaxError):
            return None

    # 类型校验
    if data_type == "dict" and not isinstance(parsed, dict):
        return None
    if data_type == "list" and not isinstance(parsed, list):
        return None
    return parsed


if __name__ == "__main__":
    # Example usage
    json_str = "{{\"extracted_triplets\": [{\"subject\": \"蔡诗萍\", \"predicate\": \"妻子\", \"object\": \"林书炜\"}, {\"subject\": \"林书炜\", \"predicate\": \"妈妈\", \"object\": \"新潮麻辣\"}, {\"subject\": \"林书炜\", \"predicate\": \"老师\", \"object\": \"新潮麻辣\"}, {\"subject\": \"林书炜\", \"predicate\": \"年龄\", \"object\": \"蔡诗萍\"}]}}"
    parsed_data = parse_json(json_str, data_type="dict")
    print(parsed_data)
