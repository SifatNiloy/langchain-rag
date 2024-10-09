from langchain_openai import OpenAIEmbeddings
from langchain.evaluation import load_evaluator
from dotenv import load_dotenv
import openai
import os

# Loading environment variables.
load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

def main():
    # Getting embedding for a word.
    embedding_function = OpenAIEmbeddings()
    vector = embedding_function.embed_query("cat")
    print(f"Vector for 'cat': {vector}")
    print("=============================================================")
    print("=============================================================")


    print(f"Vector length: {len(vector)}")

    # Compare vector of two words
    evaluator = load_evaluator("pairwise_embedding_distance")
    words = ("cat", "animal")
    x = evaluator.evaluate_string_pairs(prediction=words[0], prediction_b=words[1])
    print(f"Comparing ({words[0]}, {words[1]}): {x}")


if __name__ == "__main__":
    main()
