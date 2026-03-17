import json
import random

import sys
sys.path.append("src")  # 将上级目录添加到 sys.path 中，以便导入 llm_client 和 utils 模块

from llm_client import OpenAIClient
from utils import cosine_similarity

def read_dataset(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


class SchemaDatabase:
    def __init__(self, file_path):
        self.data = read_dataset(file_path)
        assert "object_type" in self.data[0] and "predicate" in self.data[0] and "subject_type" in self.data[0], "数据格式不正确，缺少必要字段"

        # LLM Client Initialization
        self.client = OpenAIClient()

        self.embeddings = self._build_embeddings()

    def _build_embeddings(self):
        texts = [str(entry) for entry in self.data]
        return self.client.embedding_batch(texts)

    def query(self, text, top_k):
        query_embedding = self.client.embedding(text)
        # 计算与数据库中每个条目的余弦相似度
        similarities = [cosine_similarity(query_embedding, db_emb) for db_emb in self.embeddings]
        # 获取相似度最高的 top_k 条目
        top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top_k]
        return [self.data[i] for i in top_indices]

    def query_batch(self, texts, top_k):
        query_embeddings = self.client.embedding_batch(texts)
        results = []
        for query_embedding in query_embeddings:
            similarities = [cosine_similarity(query_embedding, db_emb) for db_emb in self.embeddings]
            top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top_k]
            results.append([self.data[i] for i in top_indices])
        return results


def format_example_wo_schema(example):
    system_prompt = (
        "你是一个关系抽取助手，帮助用户从文本中提取实体关系三元组。\n请按照以下格式返回结果：{{\"extracted_triplets\": [{{\"subject\": \"实体1\", \"predicate\": \"关系\", \"object\": \"实体2\"}}]}}" +
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
    }]


def format_example_w_schema(example, schema_candidates):
    system_prompt = (
        "你是一个关系抽取助手，帮助用户从文本中提取实体关系三元组。\n请按照以下格式返回结果：{{\"extracted_triplets\": [{{\"subject\": \"实体1\", \"predicate\": \"关系\", \"object\": \"实体2\"}}]}}" +
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
    }]


if __name__ == "__main__":
    # Test the dataset loading function
    dataset = read_dataset("data/train.json")
    print(f"Loaded {len(dataset)} examples.")
    print("Sample example:", dataset[0])

    # Test the SchemaDatabase class
    if 0: # Set to 1 to run the test
        db = SchemaDatabase("data/schema.json")
        counter = 0
        recall_sum = 0

        sample_num = min(100, len(dataset))
        dataset = random.sample(dataset, sample_num)
        print(f"Evaluating on {sample_num} random examples from the dataset...")

        query_list = [example['text'] for example in dataset]
        max_predicates = max(len(example['spo_list']) for example in dataset)
        print(f"Maximum number of predicates in any example: {max_predicates}")

        top_k = 20 # max_predicates * 2  # Retrieve more candidates to increase recall
        results = db.query_batch(query_list, top_k=top_k)
        print(f"Having {len(results)} results from batch query...")

        for example, retrieved in zip(dataset, results):
            # Calculate recall based on the true triples and retrieved triples
            golden_predicate = set([spo['predicate'] for spo in example['spo_list']])
            retrieved_predicate = set([res['predicate'] for res in retrieved])
            recall = len(golden_predicate.intersection(retrieved_predicate)) / len(golden_predicate) if golden_predicate else 0
            print(f"Example {counter + 1}: Recall = {recall:.2f}")
            recall_sum += recall
            counter += 1

        print(f"\nAverage Recall: {recall_sum / counter:.2f}")

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
