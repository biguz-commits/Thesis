from tools.db_tool import query_purchases

def test_query_purchases():
    input_data = {
        "input": {
            "start_date": "2024-01-01",
            "end_date": "2025-12-31"
        }
    }
    result = query_purchases.invoke(input_data)

    assert isinstance(result, str), "The output should be a string."
    assert "Purchase ID" in result or "No purchases found" in result, "Unexpected output format."

    print("Test passed ✅")
    print("\nSample output:\n")
    print(result)

if __name__ == "__main__":
    test_query_purchases()
