from tools.vectordb_tool import query_reviews

def test_query_reviews():
    input_data = {
        "input": {
            "query_text": "fantasy books",
            "top_k": 3
        }
    }
    result = query_reviews.invoke(input_data)

    assert isinstance(result, str), "The output should be a string."
    assert result.startswith("[") and result.endswith("]"), "The output should be a JSON list."

    print("Test passed ✅")
    print("\nSample output:\n")
    print(result)

if __name__ == "__main__":
    test_query_reviews()
