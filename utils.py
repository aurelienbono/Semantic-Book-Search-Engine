import json
from openai import OpenAI
from dotenv import load_dotenv
import os
from elasticsearch import Elasticsearch

# Charger les variables d'environnement
load_dotenv()
openai_key = os.getenv('OPEN_API_KEY')

# Initialiser le client OpenAI
client = OpenAI(api_key=openai_key)

class SemanticSearchEngine:
    """Moteur de recherche sémantique utilisant Elasticsearch et OpenAI."""

    def __init__(self):
        self.client = client
        self.path_data = 'data/generate_data_book_json_version.json'
        self.index_name = 'books_index'

        # Initialiser le client Elasticsearch
        self.es_client = Elasticsearch(
            ["https://localhost:9200"],
            basic_auth=("elastic", "O7h34vCodiAuGhFMcKuf"),
            verify_certs=False
        )


        self.save_data_to_elasticsearch()

    def get_openai_embedding(self, text: str) -> list:
        """Génère un embedding pour un texte donné en utilisant OpenAI."""
        response = self.client.embeddings.create(input=text, model="text-embedding-3-small")
        embedding = response.data[0].embedding
        print("==== Generating embeddings... ====")
        return embedding

    def save_data_to_elasticsearch(self):
        """Sauvegarde les données dans Elasticsearch avec des embeddings générés."""
        if not self.es_client.indices.exists(index=self.index_name):
            self.es_client.indices.create(
                index=self.index_name,
                body={
                    "mappings": {
                        "properties": {
                            "embedding": {"type": "dense_vector", "dims": 1536},
                            "title": {"type": "text"},
                            "author": {"type": "text"},
                            "description": {"type": "text"},
                            "imageUrl": {"type": "text"},
                            "publishDate": {"type": "date"}
                        }
                    }
                }
            )

        # Charger les données JSON
        with open(self.path_data, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Boucle d’insertion avec génération d’embeddings
        for book in data:
            try:
                # Générer l'embedding
                embedding = self.get_openai_embedding(f"{book['title']} {book['description']}")
            except Exception as e:
                print(f"Erreur lors de la génération de l'embedding pour le livre ID {book['id']} : {e}")
                continue

            # Indexer dans Elasticsearch
            self.es_client.index(
                index=self.index_name,
                id=book["id"],
                body={
                    "title": book["title"],
                    "author": book["author"],
                    "description": book["description"],
                    "imageUrl": book["imageUrl"],
                    "publishDate": book["publishDate"],
                    "embedding": embedding
                }
            )

        print("Données insérées avec embeddings.")

    def search_books_by_query(self, query: str) -> list:
        """Recherche des livres similaires à une requête donnée."""
        query_embedding = self.get_openai_embedding(query)  # Obtenir l'embedding de la requête

        # Requête Elasticsearch pour la recherche des livres similaires
        query_body = {
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {"query_vector": query_embedding}
                    }
                }
            }
        }

        # Exécuter la recherche dans Elasticsearch
        results = self.es_client.search(index=self.index_name, body=query_body)

        recommendations = []
        for hit in results['hits']['hits']:
            book_info = hit['_source']
            recommendations.append({
                'title': book_info['title'],
                'author': book_info['author'],
                'description': book_info['description'],
                'score': hit['_score']
            })

        return recommendations

    def refine_recommendations_with_llm(self, query: str, results: list) -> str:
        """
        Utilise un LLM pour affiner ou enrichir les résultats de recherche.
        Retourne une recommandation reformulée par le LLM.
        """
        prompt = f"""
        Tu es un assistant de recommandation de livres.
        L'utilisateur a tapé la requête : "{query}".

        Voici les résultats de recherche initiaux :
        {json.dumps(results, indent=2)}

        Propose les meilleures recommandations basées uniquement sur les informations fournies ci-dessus.
        Ne pas inclure d'informations supplémentaires ou externes.
        Réponds en français avec des titres mis en valeur.
        """

        try:
            completion = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Tu es un assistant spécialisé en livres et recommandations de ces livres."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )

            return completion.choices[0].message.content.strip()

        except Exception as e:
            print(f"Erreur lors de l'appel au LLM : {e}")
            return "Erreur lors de l'amélioration des recommandations."
