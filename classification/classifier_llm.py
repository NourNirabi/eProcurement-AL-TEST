import pandas as pd
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Any, Optional
import json
import numpy as np
import re
import config 

class ConstructionProductClassifier:
    """
    A classifier for construction products using semantic similarity and LLM-based classification.
    Maintains a hierarchical classification tree and handles both existing and new product categories.
    """
    
    def __init__(self):
        """Initialize the classifier with embeddings model, LLM, and classification tree"""
        # Initialize embedding model and LLM
        self.embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL)
        self.llm = OllamaLLM(model=config.LLM_MODEL, temperature=0.3)
        
        # Load classification tree from CSV
        self.classification_tree = pd.read_csv(r'C:\Users\Asus\Desktop\eProcurement-AI-Test\classification\classification_tree.csv')
        
        # Prepare class data and get last used ID
        self.classes = self._prepare_classes_data()
        self.last_id = int(self.classification_tree['id'].iloc[-1])
        print(f"Last ID loaded: {self.last_id}")
        
        # Initialize prompt template and JSON parser
        self.prompt_template = self._create_focused_prompt()
        self.parser = JsonOutputParser()

    def _build_category_dict(self):
        """
        Build a dictionary mapping category IDs to their properties
        Returns:
            dict: {category_id: {name, path, parent_id, level}}
        """
        return {
            str(row['id']): {
                'name': row['name'],
                'path': row['path'],
                'parent_id': row['parent_id'],
                'level': len(str(row['path']).split('.')) if pd.notna(row['path']) else 0
            }
            for _, row in self.classification_tree.iterrows()
        }

    def _prepare_classes_data(self):
        """
        Extract and prepare class-level (Level 3) categories from the classification tree
        Returns:
            dict: {class_id: class_name} for all Level 3 categories
        """
        classes = {}
        for _, row in self.classification_tree.iterrows():
            path_parts = str(row['path']).split('.')
            if len(path_parts) >= 3 and path_parts[0] == '118814' and path_parts[1] == '119445':
                class_id = path_parts[2]
                classes[class_id] = row['name']
        print(f"Available Classes (Level 3): {classes}")
        return classes
      
    def _create_focused_prompt(self, limited_classes=None):
        """
        Create a focused prompt template for the LLM classifier
        Args:
            limited_classes (dict, optional): Subset of classes to consider. Defaults to all classes.
        Returns:
            ChatPromptTemplate: Configured prompt template
        """
        if limited_classes is None:
           limited_classes = self.classes
        class_examples = "\n".join([f"- {name} (ID: {cid})" for cid, name in limited_classes.items()])
        return ChatPromptTemplate.from_template(
      """You are an Advanced Construction Products Classification Assistant. 
      Your primary function is to analyze product attributes and classify them within our 4-level taxonomy hierarchy.

      ## Core Mission:
      1. Your first and highest-priority matching criteria is the provided **Subcategory** field.
        - Treat Subcategory as the most important signal for classification.
        - Try to map Subcategory directly to the closest available Class (Level 3) in our taxonomy.
      2. Use additional product information only to refine or confirm the Subcategory match.

      ## Classification Hierarchy Rules:
      - Path format: group_id.category_id.class_id.type_id
      - Always classify at Type level (4th level)
      - New classes/types must follow sequential ID numbering

      ## Input Analysis Workflow:
      1. Match Subcategory text to the most semantically similar existing Class (Level 3)
      2. If Subcategory does not match any existing Class with ≥80% semantic similarity:
        - Then use product technical specifications, descriptions, and metadata to decide
      3. Avoid selecting any Class that is general or unspecified unless absolutely no match exists.

      ## Strict Output Requirements (JSON ONLY):
      {{
        "class_id": "ID of BEST matching Class (Level 3)",
        "class_name": "New class name (if creating)"        
      }}

      ## Available Classes (Level 3):
      {class_examples}

      ## Current Product Data:
      {product_info}

      Important Notes:
      - Always choose the class that EXACTLY matches both the Subcategory and the product's material/application.
      - Never pick a general or 'unspecified' category if a more specific option exists.
      """
      ).partial(class_examples=class_examples)

    def _generate_new_id(self):
        """Generate a new sequential ID and update last_id counter"""
        self.last_id += 1
        return str(self.last_id)

    def classify_product(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify a product using a two-stage approach (embedding similarity + LLM)
        Args:
            product_data: Dictionary containing product attributes
        Returns:
            dict: Classification result with path, IDs, and flags
        """
        # Prepare product information string for embedding/LLM
        product_info = (
            f"Brand: {product_data.get('Brand', '')}\n"
            f"Name: {product_data.get('Product Name', '')}\n"
            f"Category: {product_data.get('Category', '')}\n"
            f"Technical Specifications: {product_data.get('Technical Specifications', '')}\n"
            f"SubCategory: {product_data.get('Subcategory', '')}\n"
            f"Short Description: {product_data.get('Short Description', '')}\n"
            f"Long Description: {product_data.get('Long Description', '')}"             
        )
        
        # Stage 1: Filter top classes using embedding similarity
        top_classes = self._filter_top_classes(product_info, top_n=3)
        print(f"Top classes suggested by embedding similarity: {top_classes}")
        
        # Stage 2: LLM classification with focused prompt
        self.prompt_template = self._create_focused_prompt(top_classes)
        response = self.llm.invoke(self.prompt_template.format(product_info=product_info))
        class_info = self._parse_response(response)
        
        # Fallback to embedding-based classification if LLM fails
        if (not class_info) or ("unspecified" in class_info.get("class_name", "").lower()):
            print("Fallback triggered due to no class or unspecified class detected.")
            return self._fallback_classification(product_data)
        
        if not class_info:
            return {"error": "Failed to classify product"}

        # Handle new class creation
        if "class_name" in class_info and class_info.get("class_id") == "Other":
            new_class_id = self._generate_new_id()
            classification_path = f"118814.119445.{new_class_id}"  # Only 3 levels (group.category.class)
            return {
                "classification_path": classification_path,
                "is_new_class": True,
                "class_id": new_class_id,
                "class_name": class_info.get("class_name", "New Category")
            }
        else:
            # Existing class - add new type (type_id)
            type_id = self._generate_new_id()
            classification_path = f"118814.119445.{class_info['class_id']}.{type_id}"
            return {
                "type_id": type_id,
                "classification_path": classification_path,
                "is_new_class": False,
                "class_name": self.classes.get(class_info['class_id'], "Unknown")
            }

    def _filter_top_classes(self, product_text, top_n=3):
        """
        Filter top classes using cosine similarity of embeddings
        Args:
            product_text: Text description of the product
            top_n: Number of top classes to return
        Returns:
            dict: {class_id: class_name} of top matching classes
        """
        class_ids = list(self.classes.keys())
        class_names = list(self.classes.values())
        product_emb = self.embeddings.embed_query(product_text)
        class_embs = [self.embeddings.embed_query(name) for name in class_names]
        sims = cosine_similarity([product_emb], class_embs)[0]
        top_indices = np.argsort(sims)[::-1][:top_n]
        return {class_ids[i]: class_names[i] for i in top_indices}

    def _parse_response(self, response: str) -> Optional[Dict]:
        """Parse LLM JSON response with error handling"""
        try:
            json_str = re.search(r'\{.*\}', response, re.DOTALL).group()
            return json.loads(json_str)
        except Exception as e:
            print(f"Error parsing JSON response: {e}")
            return None

    def _fallback_classification(self, product_data: Dict) -> Dict:
        """
        Fallback classification using only embedding similarity
        Args:
            product_data: Dictionary containing product attributes
        Returns:
            dict: Basic classification result
        """
        product_text = (
            f"{product_data.get('Product Name', '')} "
            f"{product_data.get('Short Description', '')} "
            f"{json.dumps(product_data.get('Technical Specifications', {}), ensure_ascii=False)}"
        ).lower()

        class_ids = list(self.classes.keys())
        class_names = list(self.classes.values())

        # Calculate similarity between product and all classes
        product_embedding = self.embeddings.embed_query(product_text)
        class_embeddings = [self.embeddings.embed_query(name.lower()) for name in class_names]

        similarities = cosine_similarity([product_embedding], class_embeddings)[0]
        best_index = int(np.argmax(similarities))
        best_class_id = class_ids[best_index]
        best_class_name = class_names[best_index]

        # Generate new type ID for the product
        new_type_id = self._generate_new_id()

        return {
            "type_id": new_type_id,
            "classification_path": f"118814.119445.{best_class_id}.{new_type_id}",
            "is_new_class": False,
            "class_name": best_class_name,
            "note": "Auto-generated classification (fallback mode)"
        }

    def add_to_classification(self, classification: Dict, product_name: str):
        """
        Add new classification to the classification tree
        Args:
            classification: Classification result from classify_product()
            product_name: Name of the product being classified
        """
        new_rows = []
        
        # Add new class (3 levels) if needed
        if classification['is_new_class']:
            class_row = {
                'id': classification['classification_path'].split('.')[2],
                'parent_id': '119445',
                'name': classification['class_name'],
                'path': classification['classification_path']
            }
            new_rows.append(class_row)
            print(f"Added new class: {class_row}")
        
        # Add new type if type_id exists (existing class)
        if 'type_id' in classification:
            type_row = {
                'id': classification['type_id'],
                'parent_id': classification['classification_path'].split('.')[2],
                'name': product_name,
                'path': classification['classification_path']
            }
            new_rows.append(type_row)
            print(f"Added new type: {type_row}")
        
        # Update classification tree
        self.classification_tree = pd.concat([
            self.classification_tree,
            pd.DataFrame(new_rows)
        ], ignore_index=True)
        
        # Save updates to new CSV file
        self.classification_tree.to_csv('new_classification_tree.csv', index=False)

if __name__ == "__main__":
    # Example usage
    classifier = ConstructionProductClassifier()
    
    sample_product = {
        "Subcategory": "Flooring And Coating > Industrial Flooring > Polyurethane Flooring",
        "Short Description": "2-part PUR matt coloured seal coat part of the Sika Comfortfloor® flooring range",
        "Long Description": "Sikafloor®-305 W is a two part water based, low VOC, polyurethane, coloured matt seal coat.\nSuitable for use in hot and tropical climatic conditions."
    }
    
    print("Classifying product...")
    result = classifier.classify_product(sample_product)
    print("\nClassification Result:")
    print(json.dumps(result, indent=2))
    
    if 'error' not in result:
        classifier.add_to_classification(result, sample_product.get('Product Name', 'Unnamed Product'))
        print("\nAdded to classification tree successfully!")
        
        category_dict = classifier._build_category_dict()
        # type_id exists only for existing classes
        key_id = result.get('type_id') or result.get('class_id')
        category_info = category_dict.get(key_id, {})
        print("\nCategory Details:")
        print(f"Name: {category_info.get('name', 'Unknown')}")
        print(f"Path: {category_info.get('path', 'Unknown')}")