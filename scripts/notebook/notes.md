**langchain_text_splitters**

Instead of cutting your text blindly every N words, 
it uses a prioritized list of fallback separators. 
By default, its priority list is:
1. "\n\n" (Paragraph breaks)"\n" (Line breaks)
2. " " (Spaces between words)
3. "" (Individual characters — absolute last resort)
*The Step-by-Step Splitting Process:*
Try Paragraphs first (\n\n): It looks at the text and splits it into paragraphs. If multiple paragraphs combined fit under your chunk_size (e.g., 1200 characters), it groups them together into one chunk.
Fallback to Lines (\n): If a single paragraph is too large (say, 1500 characters), it steps down to its second rule (\n) and tries to split that oversized paragraph at individual line breaks.
Fallback to Words (" "): If an individual line is still too long to fit, it steps down to its third rule (" ") and splits it at a space character between words. This ensures your text never cuts a word or sentence in half.

**What is a Sentence Transformer?**
Instead of looking for exact keyword matches (like old search engines do), a sentence transformer maps the meaning of your text into a multi-dimensional mathematical space.
*Semantic Alignment:* 
If Chunk A says "The canine barked at the mailman" and your user query is "Why is the dog making noise?", a sentence transformer understands that "canine" means "dog" and "barked" means "making noise." It generates vector numbers that place both strings right next to each other in vector space, allowing for highly accurate search results.
*Bi-Encoder Architecture:* It processes your text chunks quickly into static vector cards ahead of time, so when a user types a query, your system only has to calculate a quick similarity math score (Cosine Similarity) instead of re-reading all your text files.
**Which Model Should You Use?**
Because you are working with academic research papers on a timeline, you need a model that balances deep scientific/conceptual understanding with fast processing speed.Here are the best standard models to use, depending on your computational hardware:
1. **The Best All-Rounder: all-mpnet-base-v2**(Highly Recommended)
This is the gold standard general-purpose model provided by the library. It consistently delivers the highest quality search accuracy across diverse topics (including complex neuroscience, AI, and psychology text like your datasets).
*Embedding Dimensions:* 768 numbers per chunk.Max Text Length: 384 tokens (~300 words, matching your 1200-character chunk size perfectly).
*Verdict:* Use this if you have a modern computer or access to a GPU, as it yields the most intelligent retrieval results.
2. **The Speed King: all-MiniLM-L6-v2**
If you are running your code on an older laptop or a basic CPU without a GPU, and you notice your scripts are running slowly, switch to this model.
Embedding Dimensions: 384 numbers per chunk (half the size, saving massive hard drive space).Max Text Length: 256 tokens.
Verdict: It is roughly 5x faster than MPNET and uses a fraction of the RAM, though it is slightly less precise with complex, nuanced academic jargon.
3. **The Specialized Academic Option**: **allenai/scibert_scivocab_uncased**
If your papers are exceptionally dense with heavy biochemical formulas, neuroscience pathways, or medical terminology that normal English models struggle with, this model was explicitly pre-trained by the Allen Institute for AI on millions of academic papers.Verdict: Excellent for highly specific jargon, but requires slightly more custom configuration to initialize inside the sentence-transformers wrapper.


1.**bge-base-en-v1.5 (The Accuracy King)**
The BGE (BAAI General Embedding) model is consistently ranked near the top of the Massively Text Embedding Benchmark (MTEB) for retrieval accuracy.
The Strengths: It is exceptionally good at finding hard conceptual similarities in scientific data. If your papers use dense academic jargon, BGE will fetch the most accurate paragraphs.
The Catch (Prefixes): BGE requires "prompt instructions" to perform at its peak. When you generate embeddings for your chunks, you process them normally. But when a user asks a question, you must append a prefix string to the query:"Represent this sentence for searching relevant passages:"

**e5-base-v2 (The Speed & Balance Champion)**
Microsoft's E5 model is specifically pre-trained using a text-to-text contrastive framework, making it a favorite for production engineering pipelines.
The Strengths: It has a highly optimized layout, meaning it processes chunks faster and uses slightly less VRAM/RAM than BGE on standard systems.
The Catch (Strict Formatting): E5 strictly requires prefixes on both your dataset and your queries.When saving paper text chunks: Prepend "passage: " to every single chunk before encoding.When searching: Prepend "query: " to the user's input string.If you forget these prefixes, the retrieval accuracy drops significantly.
3. **all-mpnet-base-v2** (The Turnkey All-Rounder)
While technically older than BGE and E5, MPNET remains one of the most downloaded embedding models because of its sheer ease of use.
The Strengths: Zero formatting hygiene. You don't need prefixes, prompts, or special keyword handling for either the passages or the queries. You just pass raw strings directly into .encode(), making your script incredibly clean and less prone to logic bugs.
The Weakness: Its context window is slightly shorter (384 tokens vs 512 tokens), and its raw benchmark accuracy is slightly lower than BGE on complex retrieval tasks


Qdrant requires its point IDs to be strictly positive integers (e.g., 123) or a valid UUID string (e.g., "123e4567-e89b-12d3-a456-426614174000").
**ChromaDB (or simply Chroma)** 
It is an open-source, AI-native vector database designed specifically to store and search text embeddings.
It acts as the long-term memory for LLM applications, keeping track of your processed academic papers so you can search through them using semantic meaning instead of exact keywords.
*How It Works Under the Hood*
Traditional databases (like MySQL or Excel spreadsheets) search for data by matching exact strings or numbers. 
ChromaDB operates differently:
**Vector Storage:** It saves your text chunks alongside the dense numerical arrays (vectors) generated by your BGE embedding model.
**Nearest Neighbor Search:** When you ask a question, Chroma uses spatial math algorithms (like HNSW) to instantly calculate the geometric distance between your query vector and your stored data vectors.
**Instant Retrieval:** It returns the paragraphs that are closest in meaning to your question within milliseconds, even if your dataset contains millions of words.
**Why It Is the Industry Standard for Local Python Projects**
ChromaDB is incredibly popular for prototypes and mid-sized data science pipelines due to three key advantages:
*Zero-Configuration Setup:* It doesn't require setting up heavy external database software, hosting Docker containers, or signing up for cloud API keys.  you simply pip install chromadb and it runs entirely inside your existing Python script.
*Persistent Disk Storage:* By using chromadb.PersistentClient(), it saves your database index directly into a standard folder on your hard drive. It functions like an AI version of SQLite.
*Metadata Filtering:* It lets you attach extra tags to your text (like "source_paper": "1705.04742" or "page": 12). When querying the database later, you can filter results to look only inside specific papers or chapters.


### Qdrant vector database:
we use it to store our vector_embedding jsonl so that result or data can be fetched easily.
Qdrant is a purpose-built vector database optimized strictly for AI-driven similarity search, whereas MongoDB is a general-purpose document store that recently added vector search capabilities as a feature
**Qdrant is an open-source**, high-performance vector database. 
It is specifically designed to store, search, and manage high-dimensional vector embeddings, which are the numerical representations of data (like text, images, or audio) used in Artificial Intelligence and Machine Learning.
Why You Should Use QdrantHigh Performance:
 It is written in Rust, making it extremely fast, memory-efficient, and reliable under heavy search loads.
Advanced Filtering: You can combine vector similarity search with business logic filtering (e.g., searching for "similar images" but only "from the year 2026" or "priced under $50").
Easy Scalability: It scales seamlessly from a local Docker container during development to a distributed cloud cluster in production.
AI Ecosystem Integration: It connects natively with popular AI frameworks like OpenAI, LangChain, LlamaIndex, and Hugging Face.
**When to Choose Local Docker vs. Qdrant Cloud**
Run via Docker (docker run qdrant/qdrant)
Best for: Local development, prototyping, testing code, and offline building.
Cost: Completely free.Privacy: Data never leaves your machine.Use Qdrant Cloud (Free Tier)
Best for: Hackathons, sharing your app with others, and testing cloud deployments.
Maintenance: Zero configuration, managed upgrades, and automatic backups.
Accessibility: Your database is accessible from anywhere via an API endpoint

docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant


### Docker :
docker handle all works of qdrant-server
and we install qdrant-client
Docker is a platform that lets you package an application along with all its dependencies into a lightweight unit called a container. This ensures that your application runs the same way on your laptop, testing server, and production server.
1. Why Docker Exists
Before Docker:
"It works on my machine" was a common problem.
Different developers had different versions of:
Python
Node.js
Java
Databases
Libraries
Docker solves this by packaging everything together.
Example:
Your project requires:
Python 3.11
PostgreSQL 16
Redis
Specific libraries
Instead of asking teammates to install everything manually, you simply provide Docker files.
2. Containers vs Virtual Machines
Virtual Machine
Contains:
Full OS
Application
Libraries
Heavy and slower.
Docker Container
Contains:
Application
Dependencies only
Uses host OS kernel.
Much lighter and starts in seconds.
3. Docker Architecture
Main components:
Docker Engine
The core service that runs containers.
Docker Image
Blueprint/template.
Example:
python:3.12
node:22
ubuntu:24.04
Docker Container
Running instance of an image.
Docker Registry
Stores images.
Most common:
Docker Hub
4. Installation
Windows
Install:
Docker Desktop
Requirements:
Windows 10/11
WSL2 enabled
Check installation:
docker --version
docker compose version
Linux (Ubuntu)
sudo apt update

sudo apt install docker.io -y

sudo systemctl enable docker
sudo systemctl start docker
Add current user:
sudo usermod -aG docker $USER
Logout/Login.
Test:
docker run hello-world
5. Important Docker Concepts
Image
Think of image as:
Class → Image
Object → Container
Example:
docker pull nginx
Container
Run image:
docker run nginx
Volume
Persistent storage.
Without volume:
If container is deleted → data lost.
Network
Allows containers to communicate.
Example:
Node Container ↔ PostgreSQL Container
6. Important Terminal Commands
1. Check version
docker --version
2. Download image
docker pull ubuntu
3. Show images
docker images
Output:
REPOSITORY TAG IMAGE ID
ubuntu latest xxxx
4. Run container
docker run ubuntu
Interactive:
docker run -it ubuntu bash
5. List running containers
docker ps
All containers:
docker ps -a
6. Stop container
docker stop container_id
7. Start container
docker start container_id
8. Restart
docker restart container_id
9. Delete container
docker rm container_id
Force:
docker rm -f container_id
10. Delete image
docker rmi image_id
11. Execute command inside container
docker exec -it container_id bash
Example:
docker exec -it my-nginx bash
12. Container logs
docker logs container_id
Live logs:
docker logs -f container_id
13. Inspect details
docker inspect container_id
14. Port mapping
docker run -p 8080:80 nginx
Meaning:
Host:8080 → Container:80
Open:
http://localhost:8080
15. Naming containers
docker run --name mynginx nginx
16. Environment variables
docker run -e NODE_ENV=production myapp
17. Mount volume
docker run -v mydata:/app/data image
18. Bind local folder
docker run -v $(pwd):/app image
Windows:
docker run -v ${PWD}:/app image
19. Remove unused resources
docker system prune
Everything:
docker system prune -a
7. Dockerfile
Dockerfile is a script for creating images.
Example:
FROM node:22

WORKDIR /app

COPY package*.json ./

RUN npm install

COPY . .

EXPOSE 3000

CMD ["npm","start"]
Build:
docker build -t my-node-app .
Run:
docker run -p 3000:3000 my-node-app
8. Example Project (Node.js)
Project:
project/
│
├── app.js
├── package.json
└── Dockerfile
app.js
const express = require('express');

const app = express();

app.get('/', (req,res)=>{
    res.send('Docker App Running');
});

app.listen(3000);
package.json
{
  "dependencies": {
    "express": "^5.0.0"
  }
}
Dockerfile
FROM node:22

WORKDIR /app

COPY package*.json ./

RUN npm install

COPY . .

EXPOSE 3000

CMD ["node","app.js"]
Build:
docker build -t express-app .
Run:
docker run -p 3000:3000 express-app
Visit:
http://localhost:3000
9. Docker Compose
Compose runs multiple containers together.
Example:
Node App
PostgreSQL
Redis
docker-compose.yml
services:

  app:
    build: .
    ports:
      - "3000:3000"

    depends_on:
      - postgres

  postgres:
    image: postgres:16

    environment:
      POSTGRES_USER: admin
      POSTGRES_PASSWORD: admin
      POSTGRES_DB: testdb

    ports:
      - "5432:5432"

    volumes:
      - postgres-data:/var/lib/postgresql/data

volumes:
  postgres-data:
Run:
docker compose up
Background:
docker compose up -d
Stop:
docker compose down
10. Useful Docker Compose Commands
docker compose ps
docker compose logs
docker compose logs -f
docker compose restart
docker compose down
docker compose up --build
11. Docker Networking Example
Inside compose:
host = "postgres"
NOT:
host = "localhost"
Because containers communicate via service names.
12. Using Docker in Real Projects
MERN Stack
Containers:
Frontend → React
Backend → Node
Database → MongoDB
Redis → Cache
Nginx → Reverse Proxy
Python Project
Containers:
FastAPI
PostgreSQL
Redis
Celery
Microservices
Each service has its own container.
Example:
Auth Service
Payment Service
Email Service
Gateway
Database
Example Real Structure
myproject/

frontend/
backend/
docker-compose.yml
.env
Compose:
services:

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"

  backend:
    build: ./backend
    ports:
      - "5000:5000"

  mongo:
    image: mongo
    ports:
      - "27017:27017"
Start entire project:
docker compose up --build
| Command                   | Purpose            |
| ------------------------- | ------------------ |
| `docker images`           | List images        |
| `docker ps`               | Running containers |
| `docker ps -a`            | All containers     |
| `docker pull nginx`       | Download image     |
| `docker build -t app .`   | Build image        |
| `docker run image`        | Run container      |
| `docker exec -it id bash` | Open shell         |
| `docker logs -f id`       | Live logs          |
| `docker stop id`          | Stop               |
| `docker rm id`            | Remove container   |
| `docker rmi image`        | Remove image       |
| `docker compose up -d`    | Start services     |
| `docker compose down`     | Stop services      |
| `docker system prune -a`  | Cleanup            |

Typical Development Workflow
git clone project

cd project

docker compose up --build

docker compose logs -f

docker compose down
This is why Docker is heavily used in companies: every developer gets exactly the same environment.


### rank_bm25
popular python library used to calculate BM25 (best matching)score
which is the industry-standard mathematical algorithm for keyword text search.
While databases like Qdrant use machine learning vectors to understand the meaning of text (semantic search), rank_bm25 works by finding exact keyword matches (lexical search). 
It is the same underlying formula used by major search engines like Elasticsearch and Lucene.How the BM25 Algorithm Works:
BM25 calculates how relevant a specific search query is to a document based on three core concepts:
Term Frequency (TF): The more times a search word appears inside a document, the higher that document's score becomes.
Document Frequency (IDF): Rare words get prioritized. If your search query is "the processing error", BM25 gives very low point value to matches for "the", but massive point value to matches for "error" because "error" is a rarer, more unique keyword.
Document Length Normalization: Shorter documents that contain the keyword get a higher score than massive, multi-page documents containing the keyword. This ensures that concise, highly relevant answers rank first


The original rank_bm25 library is written in pure Python;
 it is slow on large datasets and lacks advanced text features.The best alternatives are divided by whether you want a lightweight Python library or a full database system:
1.**BM25-pt (Best Fast Python Replacement)**
 If you want to stick to a Python code library but need it to run 10x to 100x faster, use bm25-pt.
 Why it's better:
  It utilizes PyTorch under the hood. It can run calculations directly on your GPU and handles large-scale batch processing effortlessly.
 Best for:
  Data pipelines where you already use PyTorch or Hugging Face Transformers.
2. **FastEmbed by Qdrant (Best for Hybrid Search)**
   Since you are already using Qdrant, the Qdrant team built a native Python library called FastEmbed. It contains both vector generation and keyword SPLADE/BM25 indexing in a single package.
   *Why it's better:*
    It automatically formats text tokenization for sparse vectors. You can upload these keyword vectors directly into your existing Qdrant collection alongside your dense vectors. 
    This means you do not need two separate scripts; Qdrant handles both the keyword and vector searches at the same time.
    Best for: The exact pipeline you are building right now.
3.**Meilisearch (Best Open-Source Search Engine)**
 If you want a standalone, local developer search engine with an amazing user experience, Meilisearch is the modern gold standard.
 Why it's better:
  Written in Rust, it is blisteringly fast. Unlike raw BM25, it features typo-tolerance out of the box (e.g., searching "qdant" will still find "qdrant"). It also includes native support for filtering, sorting, and multi-language text.
  Best for: User-facing search bars, e-commerce apps, or documentation sites.
4. **Elasticsearch / OpenSearch (Best for Large Scale Enterprise)**
 Elasticsearch is the industry giant for keyword log parsing and enterprise search text analytics
 Why it's better: It can handle petabytes of data distributed across multiple servers. It uses an incredibly sophisticated variant of BM25 combined with boolean logic, text tokenization rules, and synonyms lists.
 mBest for: Massive production infrastructures with millions of documents


 User question: "How does attention relate to conscious access?"
        │
        ├──► Dense path: embed the question → search Qdrant → top 20 chunks by similarity
        │
        └──► Sparse path: tokenize the question → score against BM25 index → top 20 chunks by keyword overlap
                    │
                    ▼
        Both lists merged (Reciprocal Rank Fusion) → one combined ranked list
                    │
                    ▼
        Cross-encoder reranker re-scores top ~20 → picks real top 5
                    │
                    ▼
        Those 5 chunks → sent to Groq for the actual answer