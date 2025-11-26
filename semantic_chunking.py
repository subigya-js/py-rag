# How Semantic Chunking works:
# Semantic chunking breaks up long documents into meaningful pieces by finding where topics naturally change.
# Instead of cutting at random word counts, it uses AI embeddings to understand the semantic meaning of sentences.
# If one sentence talks about a certain topic and the next sentence talks about an entirely different topic, then it means we need to chunk.

# Steps in process:
# 1. Encode: Convert each sentence into embeddings (numerical vectors).
# 2. Compare: Calculate similarity score between nearby sentences.
# 3. Split: Create boundaries where similarity drops significantly.

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

tesla_text = """Tesla's Q3 Results
Tesla reported record revenue of $25.2B in Q3 2024.
The company exceeded analyst expectations by 15%.
Revenue growth was driven by strong vehicle deliveries.

Model Y Performance
The Model Y became the best-selling vehicle globally, with 350,000 units sold.
Customer satisfaction ratings reached an all-time high of 96%.
Model Y now represents 60% of Tesla's total vehicle sales.

Production Challenges
Supply chain issues caused a 12% increase in production costs.
Tesla is working to diversify its supplier base.
New manufacturing techniques are being implemented to reduce costs."""

# Semantic Chunker - groups by meaning, not structure
semantic_splitter = SemanticChunker(
    embeddings=OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=70,  # or "standard deviation"
)

chunks = semantic_splitter.split_text(tesla_text)
print("Semantic Chunking Results: ")
print("="*50)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: ({len(chunk)} chars)")
    print(f'"{chunk}')
    print()
