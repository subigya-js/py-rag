from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

tesla_text = """Tesla's Q3 Results

Tesla reported record revenue of $25.2B in Q3 2024.

Model Y Performance

The Model Y became the best-selling vehicle globally, with 350,000 units sold.

Production Challenges

Supply chain issues caused a 12% increase in production costs.

This is one very long paragraph that definitely exceeds our 100 character limit and has no double newlines inside it whatsoever making it impossible to split properly."""

# 1. CharacterTextSplitter


def character_text_splitter(text):
    print("Using CharacterTextSplitter...")
    splitter = CharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=0,
        separator=" "  # ['\n\n', '\n', '. ', '', ' ']
    )

    chunks = splitter.split_text(text)
    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- Chunk {i} ---")
        print(chunk, (f"{len(chunk)}"))


def __main__():
    character_text_splitter(tesla_text)


if __name__ == "__main__":
    __main__()

# 2. Recursive CharacterTextSplitter
print("\n" + "=" * 60)
print("2. Recursive CharacterTextSplitter Solution")
print("=" * 60)

recursive_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=100,
    chunk_overlap=0
)

chunk2 = recursive_splitter.split_text(tesla_text)
print(f"Same problem text, but with RecursiveCharacterTextSplitter:")
for i, chunk in enumerate(chunk2, 1):
    print(f"Chunk {i}: ({len(chunk)} characters)")
    print(f"{chunk}")
    print()
