from transformers import pipeline
import pandas as pd

class TableQuestionAnswering:
    def __init__(self):
        self.tqa = pipeline(task="table-question-answering", model="google/tapas-base-finetuned-wtq")
        self.table = None

    def load_table(self, table_path):
        self.table = pd.read_csv(table_path)
        self.table = self.table.astype(str)

    def answer_query(self, query):
        if self.table is None:
            raise ValueError("Table data not loaded. Call load_table() first.")

        return self.tqa(table=self.table.to_dict(orient='records'), query=query)['answer']

if __name__ == "__main__":
    tqa_instance = TableQuestionAnswering()
    tqa_instance.load_table('Model/DiseaseChatbotData.csv')
    query = 'how to prevent apple scab disease'
    answer = tqa_instance.answer_query(query)
    print(answer)
