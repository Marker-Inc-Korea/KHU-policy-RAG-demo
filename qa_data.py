import guidance
from guidance import models, gen
import pandas as pd
from autorag.data.qacreation.simple import generate_simple_qa_dataset

from dotenv import load_dotenv
load_dotenv()


def main():
    corpus_df = pd.read_parquet("../data/corpus.parquet")
    llm = models.OpenAI("gpt-3.5-turbo-16k")
    generate_simple_qa_dataset(corpus_data=corpus_df, llm=llm,
                               output_filepath="../data/qa.parquet",
                               generate_row_function=generate_qa_row)


def generate_qa_row(llm: models.Model, corpus_data_row):
    temp_llm = llm

    # make template and synthetic data with guidance
    with guidance.user():
        temp_llm += f"""
    너는 대한민국의 대학교 학생이야.
    \n
    너가 해야 할 일은 다음 2가지야.
    1. passage의 내용을 이용해서 'question'을 만들어. question 은 반드시 ?로 끝나야 해. 반드시 passage의 내용만 사용해서 question을 만들어.
    2. question에 대한 answer를 만들어. 반드시 passage의 내용만 사용해서 answer를 만들어.
    
    \n
    \n
   "passage": {corpus_data_row["contents"]}\n
   "question":
    """

    with guidance.assistant():
        temp_llm += gen('query', stop="?")
    with guidance.user():
        temp_llm += f"""
        "answer":
        """
    with guidance.assistant():
        temp_llm += gen('generation_gt')

    # add metadata in the function
    corpus_data_row["metadata"]["qa_generation"] = "simple"

    # make response dictionary
    response = {
        "query": temp_llm["query"],
        "generation_gt": temp_llm["generation_gt"]
    }
    return response


if __name__ == "__main__":
    main()
