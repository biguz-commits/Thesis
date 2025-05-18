from tools.db_tool import query_purchases

def test_query_purchases():
   
    input_data = {
        "input": {
            "start_date": "2024-01-01",
            "end_date": "2025-12-31"
        }
    }

    result = query_purchases.invoke(input_data)

    
    assert isinstance(result, str), "L'output deve essere una stringa."
    assert (
        "Price:" in result or "No purchases found" in result
    ), "Il formato dell'output non è quello atteso."

    print("✅ Test superato.")
    print("\n📄 Output di esempio:\n")
    print(result)

if __name__ == "__main__":
    test_query_purchases()
