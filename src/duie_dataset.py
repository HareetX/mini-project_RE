import json


def read_dataset(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def format_example_wo_schema(example, is_query: bool = False):
    system_prompt = (
        "你是一个关系抽取助手，帮助用户从文本中提取实体关系三元组。\n请按照以下格式返回结果：{\"extracted_triplets\": [{\"subject\": \"实体1\", \"predicate\": \"关系\", \"object\": \"实体2\"}]}" +
        "\n请从以下文本中提取实体关系三元组：")
    answer = {
        "extracted_triplets": [
            {"subject": spo['subject'], "predicate": spo['predicate'], "object": spo['object']['@value']}
            for spo in example["spo_list"]
        ]
    }
    return [{
        "role": "system",
        "content": system_prompt
    }, {
        "role": "user",
        "content": example["text"]
    }, {
        "role": "assistant",
        "content": str(answer)
    }] if not is_query else [{
        "role": "system",
        "content": system_prompt
    }, {
        "role": "user",
        "content": example["text"]
    }]


def format_example_w_schema(example, schema_candidates, is_query: bool = False):
    system_prompt = (
        "你是一个关系抽取助手，帮助用户从文本中提取实体关系三元组。\n请按照以下格式返回结果：{\"extracted_triplets\": [{\"subject\": \"实体1\", \"predicate\": \"关系\", \"object\": \"实体2\"}]}" +
        "\n以下是一些候选关系类型供参考：\n" + "\n".join([f"- {str(schema)}" for schema in schema_candidates]) +
        "\n请从以下文本中提取实体关系三元组："
    )
    answer = {
        "extracted_triplets": [
            {"subject": spo['subject'], "predicate": spo['predicate'], "object": spo['object']['@value']}
            for spo in example["spo_list"]
        ]
    }
    return [{
        "role": "system",
        "content": system_prompt
    }, {
        "role": "user",
        "content": example["text"]
    }, {
        "role": "assistant",
        "content": str(answer)
    }] if not is_query else [{
        "role": "system",
        "content": system_prompt
    }, {
        "role": "user",
        "content": example["text"]
    }]


if __name__ == "__main__":
    # Test the dataset loading function
    dataset = read_dataset("data/train.json")
    print(f"Loaded {len(dataset)} examples.")
    print("Sample example:", dataset[0])

    # Test the formatting functions
    if 1: # Set to 1 to run the test
        example = dataset[0]
        schema_candidates = read_dataset("data/schema.json")

        formatted_wo_schema = format_example_wo_schema(example)
        formatted_w_schema = format_example_w_schema(example, schema_candidates)

        print("\nFormatted example without schema candidates:")
        for msg in formatted_wo_schema:
            print(f"{msg['role']}: {msg['content']}")

        print("\nFormatted example with schema candidates:")
        for msg in formatted_w_schema:
            print(f"{msg['role']}: {msg['content']}")
