from datasets import load_dataset

def main():
  all_beaty = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", trust_remote_code=True)
  print(all_beaty["full"][0])

if __name__ == "__main__":
  main()